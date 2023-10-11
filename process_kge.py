import torch


def load_pretrain_kge(path):
    if "complex" in path:
        return load_complex_model(path)
    kge_model = torch.load(path)
    ent_embs = torch.tensor(kge_model["ent_embeddings.weight"]).cpu()
    rel_embs = torch.tensor(kge_model["rel_embeddings.weight"]).cpu()
    ent_embs.requires_grad = False
    rel_embs.requires_grad = False
    ent_dim = ent_embs.shape[1]
    rel_dim = rel_embs.shape[1]
    print(ent_dim, rel_dim)
    if ent_dim != rel_dim:
        rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)
    # print(ent_embs.shape, rel_embs.shape)
    # print(ent_embs.requires_grad, rel_embs.requires_grad)
    return ent_embs, rel_embs


def load_complex_model(path):
    kge_model = torch.load(path)
    ent_embs1 = torch.tensor(kge_model["ent_re_embeddings.weight"]).cpu()
    ent_embs2 = torch.tensor(kge_model["ent_im_embeddings.weight"]).cpu()
    rel_embs1 = torch.tensor(kge_model["rel_re_embeddings.weight"]).cpu()
    rel_embs2 = torch.tensor(kge_model["rel_im_embeddings.weight"]).cpu()
    ent_embs = torch.cat((ent_embs1, ent_embs2), dim=-1)
    rel_embs = torch.cat((rel_embs1, rel_embs2), dim=-1)
    ent_embs.requires_grad = False
    rel_embs.requires_grad = False
    ent_dim = ent_embs.shape[1]
    rel_dim = rel_embs.shape[1]
    print(ent_dim, rel_dim)
    return ent_embs, rel_embs


if __name__ == "__main__":
    load_pretrain_kge("data/CoDeX-S-complex.pth")
