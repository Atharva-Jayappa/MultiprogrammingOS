import flwr as fl
from rouge import Rouge


def calculate_rouge(reference, prediction):
    rouge = Rouge()
    scores = rouge.get_scores(reference, prediction)
    return scores[0]["rouge-1"]["f"], scores[0]["rouge-2"]["f"], scores[0]["rouge-l"]["f"]


if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8089",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
