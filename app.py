import os
import timm
import cv2
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent = os.path.dirname(os.path.abspath(__file__))

class Classifier(nn.Module):
    def __init__(self, name, *, pretrained=False, in_chans=3):
        super().__init__()
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)
        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 1)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        logits = self.fc(x)
        return logits


def inference(buffer, model):
    model.eval()
    img = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256), interpolation =cv2.INTER_LINEAR) / 255.
    img = torch.tensor(img, dtype=torch.float32).to(device).reshape([-1, 3, 256, 256])
    pred = model(img).squeeze().sigmoid().detach().cpu().numpy()
    return pred

st.header('フェイク画像判定')
st.caption('StyleGAN2で生成された画像かどうかを判定します。')

with st.spinner('モデルを読み込んでいます...'):
    model = Classifier('tf_efficientnetv2_s', pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(parent, 'efficientnetv2_s.pth'), map_location=device)['model'])


st.markdown('''
#### StyleGAN画像生成サイトの例
- [Generated Photos](https://generated.photos/faces)
- [This Person Does Not Exist](https://thispersondoesnotexist.com/)
''')
img_file_buffer = st.file_uploader(label='画像をアップロードしてください（アップロードされた画像はサーバー上には保存されません）', type=['jpg','png','jpeg','webp'])

if img_file_buffer is not None:
    img = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, width=256)
    
    with st.spinner('判定中...'):
        pred = inference(img_file_buffer, model)
    col1, col2 = st.columns(2)
    col1.metric('判定結果', 'フェイク' if pred > 0.5 else '本物')
    col2.metric('StyleGAN画像である確率', f'{pred*100:.2f}%')
