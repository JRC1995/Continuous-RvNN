
def prepare_attributes(data, args):
    if args.model_type == "NLI" or args.model_type == "Classifier":
        attributes = {"classes_num": data["classes_num"],
                      "embedding_data": data["embeddings"],
                      "vocab_len": data["vocab_len"],
                      "PAD_id": data["PAD_id"],
                      "UNK_id": data["UNK_id"],
                      "SEP_id": data["SEP_id"]}
    else:
        attributes = None

    return attributes