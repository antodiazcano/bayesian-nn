# from src.train import train
#
#
# model = train()
# x_test = torch.tensor(pd.read_csv("data/X_test.csv").values, dtype=torch.float32)
# y_test = torch.tensor(pd.read_csv("data/y_test.csv").values, dtype=torch.float32)
# y_test = torch.tensor([int(x.item()) for x in y_test])
#
#
# def test_bayesian_performance() -> None:
#    """
#    Test to check the performance of the BNN.
#    """
#
#    y_pred = model(x_test)
#    acc = (y_pred.argmax(dim=1) == y_test).float().mean().item()
#
#    assert acc >= 0.75, "Accuracy less than 70 %"
#    assert acc >= 0.80, "Accuracy less than 75 %"
#    assert acc >= 0.85, "Accuracy less than 80 %"
#
