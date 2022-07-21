# FeatureExtraction
## Installation
we use the s3prl toolkit to extract features , so you need to install s3prl first , refer to this link(https://github.com/s3prl/s3prl)
## Extract feature
The main file for extracting features is features_extraction.py, since the extracted feature dimension is 1024, the other files including the model and the trainning scripts are to facilitate you to tune your own code 
<br>Download links to the two pre-trained models 
- XLSR53(https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt)
- WavLm(https://github.com/microsoft/unilm/tree/master/wavlm)
### Load model
```python
# model_path = pretrained-model/xlsr_53.pt
  model_path = "pretrained-model/wavlm_large.pt"
# model = wav2vec2hubconf.wav2vec2_local(ckpt=model_path)
  model = wavlmhubconf.wavlm_local(ckpt=model_path)
  model.to(device)
# which layer of features to extract
  feat_layer = 16
```
### feat_extract
```python
def feat_extract(dataloader, model, device, feat_layer, save_dir,wavlm2lang):
    feat_scp_path = "{}.scp".format(os.path.join(save_dir, "feats"))
    feat_ark_path = "{}.ark".format(os.path.join(save_dir, "feats"))
    total = 0
    model.eval()
    with WriteHelper('ark,scp:' + feat_ark_path + "," + feat_scp_path) as writer:
        with torch.no_grad():
            for step, (uttid, label, wav) in enumerate(dataloader):
                wav = torch.tensor(wav).to(device=device, dtype=torch.float)
                features = model(wav)["hidden_state_{}".format(feat_layer)]
                features_ = features.squeeze(0).cpu().detach().numpy()
                writer(uttid[0],features_)
                with open(wavlm2lang,"a+") as f:
                    f.write("{} {} {}\n".format(uttid[0], label[0], features_.shape[0]))
                total += 1
    print("Total extracted features :{}".format(total))
```
