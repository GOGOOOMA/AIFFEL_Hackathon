# 🎠AIFFEL Hackathon🎠
# 🎪 TT ( Text Transformer ) 🎪 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Python&logoColor=white">

A technology that converts the text inside an image to another content or language while preserving the style.
**![](https://lh3.googleusercontent.com/Vrco3d0vuJOMdp5ChitHuu8Sk1fM2h-AYHZU7jd3bUi3gMiASQz13zAenV4OpziPyv2yI02JzUGk5ldoWExdWnorhIloWnriTzwpduoogYgTxDk5ndcU5FH_rbQzLPQmNLg4is1vrR4Z5QsF)**
## 📍 Contents
- [Description](https://github.com/GOGOOOMA/AIFFEL_Hackathon#Description)
- [Environment](https://github.com/GOGOOOMA/AIFFEL_Hackathon#Environment)
- [Reference](https://github.com/GOGOOOMA/AIFFEL_Hackathon#Reference)

## 📍 Description 
TT는 이미지 내부의 텍스트를 다른 내용으로 바꾸어주는 프로젝트로 단순히 텍스트를 바꾸는 것이 아니라 기존의 텍스트 스타일을 유치한 채 변형 시켜준다.

이 프로젝트를 실행시키기 위해 End-to-End 방식으로 Scene Text Editing을 해주는 clova ai에서 제안한 `RewriteNet`을 사용했다.

### RewriteNet [📃](https://arxiv.org/abs/2107.11041)
#### [Network]
**![](https://lh3.googleusercontent.com/IIoN02V4vgB9vktH4wKvafFRijSuqSuAlSmDFsxlbc9XvCR5mDI_JGMemJgrVGesAg0zp3FYm1MHDER39Nwt9nb413lT8cWQShl3bBCXcpyeoF538GrnzhaRuKrJOA7iaKiwsDe35LcWk0QZ94IiVw)**
- **Encoder, Generator, Recognizer, Discriminator** 총 4개의 네트워크로 구성
	- Encoder : Pre-trained ResNet-18
		- Content Encoder : Bidirectional LSTM
	- Generator : U-Net
	- Recognizer : LSTM with Attention
	- Discriminator : Discriminator of PatchGAN
- Training  phase는 두개로 구성
	- **Synthetic phase** : 합성 이미지를 이용한 훈련으로 Recognizer 부분이 있어서 이미지에서 content를 얼마나 잘 추출해 내는지를 학습 
		- Synthetic data는 [SynthTIGER](https://github.com/clovaai/synthtiger) 를 이용해서 생성
	- **Real phase** : 실제 이미지를 이용한 훈련으로 잘려진 이미지를 원본의 스타일과 얼마나 비슷하게 만들어 내는지를 훈련
		- Real data는 [ICDAR 13](https://paperswithcode.com/dataset/icdar-2013), [ICDAR 15](https://rrc.cvc.uab.es/?ch=4), [ICDAR 17](https://rrc.cvc.uab.es/?ch=8), [ICDAR 19](https://rrc.cvc.uab.es/?ch=15), [COCO-Text](https://vision.cornell.edu/se3/coco-text-2/) 를 사용 
- Inference 단계에서는 Encoder와 Generator를 사용

#### [Loss]
 **![](https://lh4.googleusercontent.com/UZ84WPhCxw1nbWBVBtGDjVY8A-rc5VF7nlBtUo9aqWHlaxNmL2pGEwXvtBmU3jwiQWUMqm1c2z3JYKTWS_-JW90aKVonE_GgsFfR6hscRoEFcXJPVN4f2gwnYHuL-YNz7H88PNnBTZHsYm5y)**

## 📍 Environment
> Python 3.9
> PyTorch 1.11


## 📍 Reference
- [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://github.com/clovaai/deep-text-recognition-benchmark)
- [StyleGAN2 with adaptive discriminator augmentation (ADA)](https://github.com/NVlabs/stylegan2-ada)
-  [SynthTIGER](https://github.com/clovaai/synthtiger)
