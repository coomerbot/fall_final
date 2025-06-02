# Challenges of GANs

GAN 被認為極難訓練的根本原因是模型的架構會產生一些數學上難以克服的困難與矛盾。

我們知道，\(G\) 的 loss \(J_G\) 可以寫成：

$$
J_G= \mathbb{E}_{x \sim p_g}[ \log(1 - D(x)) ]
$$

或：

\[
J_G= -\mathbb{E}_{x \sim p_g}[ \log(D(x)) ]
\]

而兩種寫法都有各自的問題。

第一種寫法的問題在原始的 GAN 論文裡也有提及：\(D\) 越好，\(G\) 的梯度消失越嚴重，所以 GAN 的訓練有一個技巧：我們不能將 \(D\) 訓練得太好，不然 \(G\) 會學不下去。背後的原因是因為當 \(D\) 學得很好（幾乎完美），這時的 loss \(V\) 會是：

\[
C(G)=V(D^*_G,\ G)=-\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \| p_g)
\]

而若 \(\text{supp}(p_g) \cap \text{supp}(p_{\text{data}}) = \varnothing\)，即兩個機率分佈完全不重疊的話，則根據定義有

\[
D_{\text{JS}}(p_{\text{data}} \| p_g) = \log 2,
\]

這導致 \(V\) 變成一個常數，對它取梯度是 0。要知道，在訓練初期隨機生成的 \(p_g\) 跟真實的 \(p_{\text{data}}\) 是有很大機會完全不相關的（甚至有更嚴謹的數學證明這幾乎一定發生），使得梯度消失成為一個不可忽視的問題。

第二種寫法的問題是：最小化這個 \(J_G\)，等價於最小化一個不合理的距離衡量，造成梯度不穩定與嚴重的 mode collapse。

首先，\(D_{\text{KL}}(p_{g} \| p_{\text{data}})\) 代入最優 \(D^*_G\)，是：

\[
D_{\text{KL}}(p_{g} \| p_{\text{data}}) = \mathbb{E}_{x \sim p_g} \left[ \log\left(\frac{p_g}{p_{\text{data}}}\right) \right] = \mathbb{E}_{x \sim p_g} \left[ \log\left( \frac{1 - D^*_G(x)}{D^*_G(x)} \right) \right]
\]

\[
= \mathbb{E}_{x \sim p_g} [\log(1 - D^*_G(x))] - \mathbb{E}_{x \sim p_g} [\log(D^*_G(x))]
\]

因此，得到第二種寫法其實是：

\[
J_G = - \mathbb{E}_{x \sim p_g} [\log(D^*_G(x))] = D_{\text{KL}}(p_{g} \| p_{\text{data}}) - \mathbb{E}_{x \sim p_g} [\log(1 - D^*_G(x))]
\]

\[
= D_{\text{KL}}(p_{g} \| p_{\text{data}}) - 2 \cdot D_{\text{JS}}(p_{\text{data}} \| p_g) + \log 4 + \mathbb{E}_{x \sim p_{\text{data}}} [\log D^*_G(x)]
\]

等價於最小化：

\[
D_{\text{KL}}(p_{g} \| p_{\text{data}}) - 2 \cdot D_{\text{JS}}(p_{\text{data}} \| p_g)
\]

這個最小化目標有兩個嚴重的問題。第一個問題是它要最小化生成分佈與真實分佈的 KL 散度，卻又要最大化兩者的 JS 散度，同時拉近跟推遠，這在直觀上是矛盾且荒謬的。

第二個問題是 \(D_{\text{KL}}(p_{g} \| p_{\text{data}})\) 會給出兩種不平等的懲罰：

1. \(p_g(x) = 0, \ p_{\text{data}}(x) > 0\)：\(G\) 沒能涵蓋所有真實樣本，生成缺乏多樣性。

\[
p_g(x) \to 0, \quad p_{\text{data}}(x) \to 1, \quad p_g(x) \cdot \frac{p_g(x)}{p_{\text{data}}(x)} \to 0, \quad D_{\text{KL}}(p_{g} \| p_{\text{data}}) \to 0
\]

2. \(p_g(x) > 0, \ p_{\text{data}}(x) = 0\)：\(G\) 亂編出根本不存在的樣本，生成缺乏準確性。

\[
p_g(x) \to 1, \quad p_{\text{data}}(x) \to 0, \quad p_g(x) \cdot \frac{p_g(x)}{p_{\text{data}}(x)} \to +\infty, \quad D_{\text{KL}}(p_{g} \| p_{\text{data}}) \to +\infty
\]

KL 散度對「缺乏多樣性」的錯誤非常寬容，但對「缺乏準確性」的錯誤非常嚴厲，這將導致 \(G\) 變得保守，寧可多生成一些重複但安全的樣本，也不願意去生成有多樣性的樣本，造成 mode collapse。

在實作中很容易發生的狀況是當 \(G\) 發現生成某種圖片能騙過 \(D\) 後，不管我們怎麼改變 noise，它都會一直生成那張圖片。

---

# Wasserstein Distance

Wasserstein 距離又稱為 EM 距離 (earth-mover distance)，是 KL、JS 散度之外，一種描述兩個分佈之間的距離的方式，它的定義為：

\[
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]
\]

解釋如下：

\(\Pi(p_{\text{data}}, p_g)\) 是 \(p_{\text{data}}\) 和 \(p_g\) 組合起來的所有可能的聯合分布的集合，換句話說，\(\Pi(p_{\text{data}}, p_g)\) 中每一個分布的邊緣分佈都是 \(p_{\text{data}}\) 和 \(p_g\)。

對於每個可能的聯合分布 \(\gamma\) 而言，可以從中採樣 \((x,y) \sim \gamma\) 得到一個真實樣本 \(x\) 和一個生成樣本 \(y\)，並計算這對樣本的距離 \(\|x - y\|\)，因此可以計算該聯合分佈 \(\gamma\) 下樣本對距離的期望值 \(\mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|]\)。

在所有可能的聯合分布中能夠對這個期望值取到的下界

\[
\inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|]
\]

即為 Wasserstein 距離。

直觀上，可以將 \(\mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|]\) 理解為在 \(\gamma\) 這個「路徑規劃」下，將 \(p_r\) 這堆「沙土」搬運至 \(p_g\) 位置所需的「消耗」，而 \(W(p_r, p_g)\) 則是「最優路徑規劃」下的「最小消耗」，因此被稱為 earth-mover distance。
