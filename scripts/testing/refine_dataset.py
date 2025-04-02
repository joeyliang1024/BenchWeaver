import asyncio
import atexit
import os
import signal
from concurrent.futures import Future
from functools import wraps
from pathlib import Path
from types import FrameType
from typing import Callable, Coroutine

import fire
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset, load_from_disk
from openai import AsyncOpenAI
from tqdm import tqdm


def async_main(fn: Callable[..., Coroutine]):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(fn(*args, **kwargs))
        except KeyboardInterrupt: ...
    return wrapper


@async_main
async def main(
    dataset_kwargs: dict = {
        # 'path': 'wikimedia/wikipedia', 'name': '20231101.zh'
        'path': 'BAAI/CCI3-HQ'
    },
    # output_dir: str | Path = 'data/pt/wikipedia/20231101.zhtw.14b',
    output_dir: str | Path = 'data/pt/cci3-hq/qwen-14b-zhtw',
    pre_processed_data_dir: str | Path | None = 'data/pt/cci3-hq/chunked',
    api_url: str = 'http://localhost:7000/v1',
    model: str = 'Qwen/Qwen2.5-14B-Instruct',
    system_prompt: str | None = (
        '你是一個不喜歡使用簡體中文的台灣人\n'
        '你需要將使用者提供的簡體中文文本轉換成繁體中文\n'
        '其他非簡體中文的部分不需要進行翻譯\n'
        '請適時的將中國用語轉換成台灣用語\n'
        '若文本的格式或語意有問題，可以適當的調整內容\n'
    ),
    max_length: int = 10240, # 10240 14336 65536
    batch_size: int = 1000,
    max_concurrency: int = 32 * 32,
    num_proc: int = len(os.sched_getaffinity(0))
):
    output_dir = Path(output_dir)
    if pre_processed_data_dir is not None:
        pre_processed_data_dir = Path(pre_processed_data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        base_url=api_url,
        timeout=None,
        max_retries=10
    )

    if pre_processed_data_dir is not None and pre_processed_data_dir.is_dir():
        dataset = load_from_disk(pre_processed_data_dir)
    else:
        dataset_kwargs.setdefault('split', 'train')
        dataset_kwargs.setdefault('num_proc', num_proc)
        dataset = load_dataset(**dataset_kwargs)

        dataset = dataset.map(
            chunk_text,
            fn_kwargs=dict(
                max_length=max_length
            ),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc='Chunking text'
        )

        dataset = dataset.sort('score', reverse=True)

        if pre_processed_data_dir is not None:
            dataset.save_to_disk(pre_processed_data_dir, num_proc=num_proc)
    
    # dataset = dataset.select(range(24))

    parquet_writer = None
    initial_examples = 0
    initial_charachers = 0
    skipping_ids = set()
    parquet_file_idx = 0
    for p in output_dir.glob('part-*.parquet'):
        with pq.ParquetFile(p) as f:
            for batch in f.iter_batches(batch_size, columns=['id', 'original_length']):
                initial_examples += len(batch)
                initial_charachers += sum(batch['original_length'].to_pylist())
                skipping_ids.update(batch['id'].to_pylist())
        parquet_file_idx += 1

    total_chars = sum(dataset['length'])
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_bar = tqdm(
        total=len(dataset),
        initial=initial_examples,
        unit=' example',
        dynamic_ncols=True
    )
    char_progress_bar = tqdm(
        total=total_chars,
        initial=initial_charachers,
        unit=' char',
        dynamic_ncols=True
    )
    example_buffer = []

    def write_examples():
        nonlocal parquet_writer, example_buffer

        if example_buffer:
            record_batch = pa.RecordBatch.from_pylist(example_buffer)
            parquet_writer.write_batch(record_batch)
            example_buffer.clear()
    
    def finalize():
        nonlocal parquet_writer

        write_examples()

        if parquet_writer is not None:
            parquet_writer.close()

    atexit.register(finalize)

    def done_callback(f: Future[dict]) -> None:
        nonlocal parquet_writer, semaphore, char_progress_bar, example_buffer

        try:
            r = f.result()

            if parquet_writer is None:
                schema = pa.RecordBatch.from_pylist([r]).schema
                parquet_writer = pq.ParquetWriter(output_dir / f'part-{parquet_file_idx:05d}.parquet', schema)

            example_buffer.append(r)

            if len(example_buffer) >= batch_size:
                write_examples()

            progress_bar.update()
            char_progress_bar.update(r['original_length'])
        except asyncio.CancelledError: ...
        
        semaphore.release()

    tasks = []
    try:
        for example in dataset:
            if example['id'] in skipping_ids:
                continue

            await semaphore.acquire()
            c = get_response(
                client=client,
                model=model,
                system_prompt=system_prompt,
                example=example
            )
            t = asyncio.create_task(c)
            t.add_done_callback(done_callback)
            tasks.append(t)
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        progress_bar.close()
        char_progress_bar.close()

        print('Canceling tasks')

        for t in tasks:
            t.cancel()

    progress_bar.close()
    char_progress_bar.close()
    
    finalize()
    
    atexit.unregister(finalize)

    print('Done')


def chunk_text(
    batch: dict[str, list],
    max_length: int
) -> dict[str, list]:
    new_batch = {k: [] for k in batch.keys()}
    new_batch |= {k: [] for k in ['original_id', 'length']}

    modified_keys = {'text', 'id', 'original_id', 'length'}
    unmodified_keys = new_batch.keys() - modified_keys
    
    for batch_idx in range(len(batch['text'])):
        id_ = batch['id'][batch_idx]
        text = batch['text'][batch_idx]

        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        num_chunks = len(chunks)
        new_batch['text'] += chunks
        new_batch['original_id'] += [id_] * num_chunks
        new_batch['id'] += [f'{id_}_{i}' for i in range(num_chunks)]
        new_batch['length'] += [len(s) for s in chunks]

        for k in unmodified_keys:
            new_batch[k] += [batch[k][batch_idx]] * num_chunks

    return new_batch


async def get_response(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    example: dict
) -> dict:
    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    
    messages.append({'role': 'user', 'content': example['text']})
    
    c = await client.chat.completions.create(
        messages=messages,
        model=model
    )
    response = c.choices[0].message.content
    
    example['original_text'] = example.pop('text')
    example['original_length'] = example.pop('length')
    example['text'] = response
    example['length'] = len(response)
    return example


def handle_sigterm(signum: int, frame: FrameType) -> None:
    signal.raise_signal(signal.SIGINT)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, handle_sigterm)

    fire.Fire(main)
