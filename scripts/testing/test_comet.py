from comet import download_model, load_from_checkpoint
from comet.models.utils import Prediction

if __name__ == "__main__":
    model_path = download_model("Unbabel/wmt20-comet-qe-da")
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src": "The output signal provides constant sync so the display never glitches.",
            "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
        },
        {
            "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
            "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
        },
        {
            "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
            "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
        }
    ]
    model_output: Prediction = model.predict(data, batch_size=8, gpus=2, devices=[1,2])
    print (model_output)
    print(model_output.scores)
    print(model_output.system_score)
    # Prediction([('scores', [0.30484065413475037, 0.23435932397842407, 0.6128204464912415]), ('system_score', 0.384006808201472)])