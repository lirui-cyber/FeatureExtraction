import argparse
from model import *
from sklearn.preprocessing import LabelEncoder
import s3prl.upstream.wavlm.hubconf as wavlmhubconf
import s3prl.upstream.wav2vec2.hubconf as wav2vec2hubconf
import torch
import json
from data_load import *
from kaldiio import  WriteHelper
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
def wav_lang_extract(wavscp, utt2lang):
    with open(wavscp, 'r') as f:
        lines_wav = f.readlines()
    audio_list = [x.split()[-1].strip() for x in lines_wav]
    name_list = [x.split()[0].strip() for x in lines_wav]
    with open(utt2lang, 'r') as f:
        lines_utt = f.readlines()
    label_list = [x.split()[-1].strip().replace('-', '') for x in lines_utt]
    return audio_list, label_list, name_list


def prepare_data(wav_scp_train,utt2lang_train):
    audio_train, labels_train, name_list = wav_lang_extract(wav_scp_train, utt2lang_train)
    le = LabelEncoder()
    labels_train_index = le.fit_transform(labels_train)
    train_set = RawFeatures3(name_list, audio_train, labels_train_index)
    trainloader = DataLoader(dataset=train_set,
                             batch_size=1,
                             pin_memory=True,
                             num_workers=1)
    return trainloader
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

def main():

    device = torch.device("cuda")
    # model_path = pretrained-model/xlsr_53.pt
    model_path = "pretrained-model/wavlm_large.pt"
    # model = wav2vec2hubconf.wav2vec2_local(ckpt=model_path)
    model = wavlmhubconf.wavlm_local(ckpt=model_path)
    model.to(device)
    feat_layer = 16

    # prepare data
    wav_scp_train = "wav.scp"
    utt2lang_train = "utt2lang"
    trainloader = prepare_data(wav_scp_train, utt2lang_train)
    save_dir = "wavlm_large_" + str(feat_layer) + "_layer"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    train_txt = "wavlm_large_pretrained_model.txt"
    # feat_extract
    feat_extract(trainloader, model, device, feat_layer, save_dir,train_txt)

if __name__ == '__main__':
    main()
