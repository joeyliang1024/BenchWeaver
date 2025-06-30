import os
import json
import pandas as pd
import uuid
import re
import gradio as gr
import threading
import time  # For simulating work

# --- Configuration ---
OPQA_DIR = "/work/u5110390/BenchWeaver/score/main_pipeline/gsm8k/en"
MCQA_DIR = "/work/u5110390/BenchWeaver/score/main_pipeline/tmmluplus/zh-tw"
OUTPUT_DIR = "/work/u5110390/BenchWeaver/logs/annotations"
ANNOTATE_COUNTS = 50
# (Your dummy data setup code omitted for brevity)

def load_check_results(json_path: str):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except:
        return {}

def load_check_prompts(json_path: str):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except:
        return {}

def parse_bool_score(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r'\b(true|false|unknown)\b', text.lower())
    if m:
        if m.group(1) == "true":    
            return True
        elif m.group(1) == "false":
            return False
        else:
            return "Unknown"
    return ""

def merge_check_results_and_prompts(check_results, check_prompts):
    merged = []
    for key in check_results:
        if key in check_prompts and len(check_results[key]) == len(check_prompts[key]):
            for result, prompt in zip(check_results[key], check_prompts[key]):
                merged.append({
                    "check_result": parse_bool_score(result),
                    "check_prompt": prompt[0]['content'],
                    "subject": key,
                    "ID": str(uuid.uuid4()),
                    **{f"annotation{i+1}": None for i in range(3)}
                })
    return merged

def get_combined_dataframe():
    opqa_df = pd.DataFrame(merge_check_results_and_prompts(
        load_check_results(os.path.join(OPQA_DIR, "check_results.json")),
        load_check_prompts(os.path.join(OPQA_DIR, "checked_prompts.json"))
    ))
    opqa_df = opqa_df.head(ANNOTATE_COUNTS)
    mcqa_df = pd.DataFrame(merge_check_results_and_prompts(
        load_check_results(os.path.join(MCQA_DIR, "check_results.json")),
        load_check_prompts(os.path.join(MCQA_DIR, "checked_prompts.json"))
    ))
    mcqa_df = mcqa_df.head(ANNOTATE_COUNTS)
    if opqa_df.empty and mcqa_df.empty:
        return pd.DataFrame([{
            "check_result": "dummy", "check_prompt":"No prompt available.",
            "subject":"none", "ID":"dummy-id",
            "annotation1": None, "annotation2": None, "annotation3": None
        }])
    df = pd.concat([opqa_df, mcqa_df], ignore_index=True).drop_duplicates("ID")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "combined_check.csv"), index=False)
    return df

df = get_combined_dataframe()
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3"]
ANNOTATION_COLS = ["annotation1", "annotation2", "annotation3"]

def get_next_index(df, col):
    un = df[df[col].isnull()]
    return un.index[0] if not un.empty else -1

def get_progress(df, col):
    tot = len(df)
    done = df[col].notna().sum()
    return f"Annotations Completed: {done} / {tot}", done 

def load_item(df, annotator, progress=gr.Slider()):
    col = ANNOTATION_COLS[ANNOTATORS.index(annotator)]
    idx = get_next_index(df, col)
    text, frac = get_progress(df, col)
    if idx == -1:
        return -1, "🎉 All done!", text, gr.update(value=frac)
    return idx, df.at[idx, "check_prompt"], text, gr.update(value=frac)

def annotate_and_next(df, idx, annotator, annotation_value, progress=gr.Slider()):
    col = ANNOTATION_COLS[ANNOTATORS.index(annotator)]

    if idx == -1:
        return df, -1, "🎉 Completed!", "🎉 Completed!", gr.update(value=1.0)

    df.at[idx, col] = True if annotation_value == 1 else False 
    df.to_csv(os.path.join(OUTPUT_DIR, "combined_check.csv"), index=False)

    # Unpack all 4 values from load_item
    next_idx, next_prompt, progress_text, progress_bar = load_item(df, annotator)
    return df, next_idx, next_prompt, progress_text, progress_bar  # ✅ now 5 values

def analyze_data(df, progress=gr.Progress(track_tqdm=True)):
    res = "### Annotation Analysis\n"
    if os.path.exists(os.path.join(OUTPUT_DIR, "['All Annotators']_annotated.csv")):
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "['All Annotators']_annotated.csv"))
        all_annotated = True
    else:
        df = df
        all_annotated = False
        
    for annotator, col in zip(ANNOTATORS, ANNOTATION_COLS):
        if all_annotated:
            print(f"All annotated. Analyzing {annotator}...")
            annotated = df[col].notna().sum()
            trues = int((df[col] == True).sum())
            falses = int((df[col] == False).sum())
        else:
            # check by if df exists
            if os.path.exists(os.path.join(OUTPUT_DIR, f"['{annotator}']_annotated.csv")):
                df = pd.read_csv(os.path.join(OUTPUT_DIR, f"['{annotator}']_annotated.csv"))
                print(f"Loaded {annotator} annotations.")
                annotated = df[col].notna().sum()
                trues = int((df[col] == True).sum())
                falses = int((df[col] == False).sum())
            else:
                annotated = 0
                trues = 0
                falses = 0
        res += f"- **{annotator}**: {annotated} items ({trues} True, {falses} False)\n"
        time.sleep(0.5)
    res += "\nGenerating final report...\n"
    for _ in progress.tqdm(range(5), desc="Finalizing"):
        time.sleep(0.1)
    return res

with gr.Blocks(title="Annotation Interface") as demo: # theme=gr.themes.Soft(), 
    dataframe_state = gr.State(df)
    current_index = gr.State(-1)
    gr.Markdown("# Annotation Interface for QA Evaluation")
    gr.Markdown("Select your annotator, read the prompt, and classify it.")

    with gr.Row():
        with gr.Column(scale=1):
            annotator_select = gr.Dropdown(ANNOTATORS, label="Annotator ID")
            progress_text = gr.Textbox(label="Progress", interactive=False)
            progress_bar = gr.Slider(1, (2 * ANNOTATE_COUNTS), value=0.0, step=1, label="Progress", interactive=False)

            with gr.Accordion("Advanced Analysis", open=False):
                analyze_btn = gr.Button("Analyze")
                analysis_out = gr.Markdown()
            with gr.Accordion("Manual Save", open=False):
                save_group = gr.CheckboxGroup(["All Annotators"] + ANNOTATORS, value=["All Annotators"])
                save_btn = gr.Button("Save")
                save_status = gr.Textbox(interactive=False)

        with gr.Column(scale=3):
            prompt_box = gr.Textbox(lines=15, interactive=False, label="Check Prompt")
            true_btn = gr.Button("✅ True")
            false_btn = gr.Button("❌ False")

    # Events
    annotator_select.change(
        load_item, [dataframe_state, annotator_select], [current_index, prompt_box, progress_text, progress_bar]
    )
    true_btn.click(
        annotate_and_next,
        [dataframe_state, current_index, annotator_select, gr.Number(value=1, visible=False)],
        [dataframe_state, current_index, prompt_box, progress_text, progress_bar]
    )
    false_btn.click(
        annotate_and_next,
        [dataframe_state, current_index, annotator_select, gr.Number(value=0, visible=False)],
        [dataframe_state, current_index, prompt_box, progress_text, progress_bar]
    )
    analyze_btn.click(analyze_data, dataframe_state, analysis_out)
    save_btn.click(lambda df, sel: (df.to_csv(os.path.join(OUTPUT_DIR, f"{sel}_annotated.csv"), index=False),
                                     f"Saved for {sel}"),
                   [dataframe_state, save_group], save_status)

    demo.load(load_item, [dataframe_state, annotator_select], [current_index, prompt_box, progress_text, progress_bar])

demo.launch(
    server_name="0.0.0.0", server_port=5002,
    ssl_certfile="/etc/ssl/ccs.twcc.ai/fullchain.pem",
    ssl_keyfile="/etc/ssl/ccs.twcc.ai/privkey.pem",
    ssl_verify=False
)
