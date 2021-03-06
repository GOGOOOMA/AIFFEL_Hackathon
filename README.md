# ๐ AIFFEL Hackathon๐ 
# ๐ช TT ( Text Transformer ) ๐ช 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Python&logoColor=white">

A technology that converts the text inside an image to another content or language while preserving the style.
**![](https://lh3.googleusercontent.com/Vrco3d0vuJOMdp5ChitHuu8Sk1fM2h-AYHZU7jd3bUi3gMiASQz13zAenV4OpziPyv2yI02JzUGk5ldoWExdWnorhIloWnriTzwpduoogYgTxDk5ndcU5FH_rbQzLPQmNLg4is1vrR4Z5QsF)**
## ๐ Contents
- [Description](https://github.com/GOGOOOMA/AIFFEL_Hackathon#-Description)
- [Environment](https://github.com/GOGOOOMA/AIFFEL_Hackathon#-environment)
- [Reference](https://github.com/GOGOOOMA/AIFFEL_Hackathon#-Reference)

## ๐ Description 
TT๋ ์ด๋ฏธ์ง ๋ด๋ถ์ ํ์คํธ๋ฅผ ๋ค๋ฅธ ๋ด์ฉ์ผ๋ก ๋ฐ๊พธ์ด์ฃผ๋ ํ๋ก์ ํธ๋ก ๋จ์ํ ํ์คํธ๋ฅผ ๋ฐ๊พธ๋ ๊ฒ์ด ์๋๋ผ ๊ธฐ์กด์ ํ์คํธ ์คํ์ผ์ ์ ์นํ ์ฑ ๋ณํ ์์ผ์ค๋ค.

์ด ํ๋ก์ ํธ๋ฅผ ์คํ์ํค๊ธฐ ์ํด End-to-End ๋ฐฉ์์ผ๋ก Scene Text Editing์ ํด์ฃผ๋ clova ai์์ ์ ์ํ `RewriteNet`์ ์ฌ์ฉํ๋ค.

### RewriteNet [๐](https://arxiv.org/abs/2107.11041)
#### [Network]
**![](https://lh3.googleusercontent.com/IIoN02V4vgB9vktH4wKvafFRijSuqSuAlSmDFsxlbc9XvCR5mDI_JGMemJgrVGesAg0zp3FYm1MHDER39Nwt9nb413lT8cWQShl3bBCXcpyeoF538GrnzhaRuKrJOA7iaKiwsDe35LcWk0QZ94IiVw)**
- **Encoder, Generator, Recognizer, Discriminator** ์ด 4๊ฐ์ ๋คํธ์ํฌ๋ก ๊ตฌ์ฑ
	- Encoder : Pre-trained ResNet-18
		- Content Encoder : Bidirectional LSTM
	- Generator : U-Net
	- Recognizer : LSTM with Attention
	- Discriminator : Discriminator of PatchGAN
- Training  phase๋ ๋๊ฐ๋ก ๊ตฌ์ฑ
	- **Synthetic phase** : ํฉ์ฑ ์ด๋ฏธ์ง๋ฅผ ์ด์ฉํ ํ๋ จ์ผ๋ก Recognizer ๋ถ๋ถ์ด ์์ด์ ์ด๋ฏธ์ง์์ content๋ฅผ ์ผ๋ง๋ ์ ์ถ์ถํด ๋ด๋์ง๋ฅผ ํ์ต 
		- Synthetic data๋ [SynthTIGER](https://github.com/clovaai/synthtiger) ๋ฅผ ์ด์ฉํด์ ์์ฑ
	- **Real phase** : ์ค์  ์ด๋ฏธ์ง๋ฅผ ์ด์ฉํ ํ๋ จ์ผ๋ก ์๋ ค์ง ์ด๋ฏธ์ง๋ฅผ ์๋ณธ์ ์คํ์ผ๊ณผ ์ผ๋ง๋ ๋น์ทํ๊ฒ ๋ง๋ค์ด ๋ด๋์ง๋ฅผ ํ๋ จ
		- Real data๋ [ICDAR 13](https://rrc.cvc.uab.es/?ch=2&com=introduction), [ICDAR 15](https://rrc.cvc.uab.es/?ch=4), [ICDAR 17](https://rrc.cvc.uab.es/?ch=8), [ICDAR 19](https://rrc.cvc.uab.es/?ch=15), [COCO-Text](https://vision.cornell.edu/se3/coco-text-2/) ๋ฅผ ์ฌ์ฉ 
- Inference ๋จ๊ณ์์๋ Encoder์ Generator๋ฅผ ์ฌ์ฉ

#### [Loss]
 **![](https://lh4.googleusercontent.com/UZ84WPhCxw1nbWBVBtGDjVY8A-rc5VF7nlBtUo9aqWHlaxNmL2pGEwXvtBmU3jwiQWUMqm1c2z3JYKTWS_-JW90aKVonE_GgsFfR6hscRoEFcXJPVN4f2gwnYHuL-YNz7H88PNnBTZHsYm5y)**

## ๐ Environment
> Python 3.9  
> PyTorch 1.11


## ๐ Reference
- [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://github.com/clovaai/deep-text-recognition-benchmark)
- [StyleGAN2 with adaptive discriminator augmentation (ADA)](https://github.com/NVlabs/stylegan2-ada)
-  [SynthTIGER](https://github.com/clovaai/synthtiger)
