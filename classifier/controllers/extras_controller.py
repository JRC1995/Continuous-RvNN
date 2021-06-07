
def extras_fn(data, args):

    extras = {}
    if "idx2labels" in data:
        extras["idx2labels"] = data["idx2labels"]

    return extras