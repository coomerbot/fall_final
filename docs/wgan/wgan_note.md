# Challenges of GANs

GAN 被認為極難訓練的根本原因是模型的架構會產生一些數學上難以克服的困難與矛盾。

我們知道，$`G`$ 的 loss $`J_G`$ 可以寫成：

```math
J_G= \mathbb{E}_{x \sim p_g}[ \log(1 - D(x) ]
```

或：

```math
J_G= -\mathbb{E}_{x \sim p_g}[ \log(D(x)) ]
```

而兩種寫法都有各自的問題。

第一種寫法的問題在原始的 GAN 論文裡也有提及：$`D`$ 越好，$`G`$ 的梯度消失越嚴重，所以 GAN 的訓練有一個技巧：我們不能將 $`D`$ 訓練的太好，不然 $`G`$ 會學不下去。背後的原因是因為當 $`D`$ 學得很好（幾乎完美），這時的loss $`V`$ 會是：

```math
C(G)=V(D^*_G,\ G)=-\log 4+2\cdot D_{\text{JS}}(p_{\text{data}}||p_g)
```

而若 $`\text{supp}(p_g) \cap\text{supp}(p_{\text{data}})=0`$，即兩個機率分佈完全不重疊的話，則根據定義有 $`D_{\text{JS}}(p_{\text{data}}||p_g)=\log 2`$，這導致 $`V`$ 變成一個常數，對它取梯度是 $`0`$。要知道，在訓練初期隨機生成的 $`p_g`$ 跟真實的 $`p_{\text{data}}`$ 是有很大機會完全不相關的（甚至有更嚴謹的數學證明這幾乎一定發生），使得梯度消失成為一個不可忽視的問題。

第二種寫法的問題是：最小化這個 $`J_G`$，等價於最小化一個不合理的距離衡量，造成梯度不穩定與嚴重的mode collapse。

首先，$`D_{\text{KL}}(p_{g}||p_{\text{data}})`$ 代入最優 $`D^*_G`$，是：

```math
D_{\text{KL}}(p_{g}||p_{\text{data}})=\mathbb{E}_{x\sim p_g}[\log(p_g/p_{\text{data}})]=\mathbb{E}_{x\sim p_g}[\log(1-D^*_G(x)/D^*_G(x))]
```
```math
=\mathbb{E}_{x\sim p_g}[\log(1-D^*_G(x))]-\mathbb{E}_{x\sim p_g}[\log(D^*_G(x))]
```

因此，得到第二種寫法其實是：

```math
J_G=-\mathbb{E}_{x\sim p_g}[\log(D^*_G(x))]=
```
