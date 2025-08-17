\section{Methodology}

We define the domain generalization task as follows: given $K$ source domains $ D_S = \{D_1,...,D_K\}$ as the training set, where the $i$-th domain $D_i$ have $N_i$ image-label sample pairs $\{(x_{j}^{i}, y_{j}^{i})\}_{j=1}^{N_i}$. The target is to learn a model from multiple source domains that can be generalized to the target domains $D_T$ with unknown distributions.

In this paper, to relieve the problem of network performance degradation caused by domain shifts, we propose a frequency-domain-based feature disentanglement and interaction (FFDI) framework. The overall network structure of our method is illustrated in Fig.~\ref{fig:2}(a). The feature extractor $E(\cdot)$ maps the input image to the embedding feature $f_E$. The disentangler $D(\cdot)$ is composed of two parallel convolutional layers and is responsible for disentangling $f_E$ into high-frequency features $f_H$ and low-frequency features $f_L$. The image reconstructor $R(\cdot)\in \{R_H(\cdot), R_L(\cdot)\}$ aims to recover high-pass (or low-pass) filtered images from $f_H$ (or $f_L$). Hence $D(\cdot)$ and $R(\cdot)$ can be 
viewed as the encoder-decoder structure in Convolution Auto-Encoders (CAE). To effectively utilize helpful information in high- and low-frequency features, we design a  information interaction mechanism $IMM(\cdot)$ for the fusion of $f_H$ and $f_L$. After that, we use three classifiers $C_{A_H}(\cdot)$, $C_{A_L}(\cdot)$, $C_I(\cdot)$ corresponding to high-frequency features, low-frequency features, and fused features respectively to predict the target category. Note that $C_{A_H}(\cdot)$ and $C_{A_L}(\cdot)$ only work as auxiliary classifiers to enhance the discriminative information in $f_H$ and $f_L$ and will be discard during inference. Furthermore, we propose a frequency-domain-based data augmentation method named FDAG, which can enhance the robustness of the feature disentanglement. Next, we will describe each module of FFDI in detail.

\subsection{Feature Disentanglement}
\label{sec3.1}
We use CAE to extract high- and low-frequency features of the image. First of all, we transform each channel of the RGB image $I$ to the frequency domain space by using the Fourier transformation, which is formulated as:
%
\begin{align}
\label{Eq.1}
    F(u,v) = \sum_{a=0}^{A-1} \sum_{b=0}^{B-1} x(a,b) e^{-j2\pi(au/A + bv/B)},
\end{align}
%
and for simplified we donate Fourier transformation as $F(\cdot)$ and use the symbol $F^{-}(\cdot)$ to represent the inverse Fourier transformation. Further, we denote with $M$ a mask, whose value is zero except for the center region:

\begin{equation}
\label{eq6}
M={\left\{
\begin{aligned}
1 & , (u,v) \in [c_x-r:c_x+r, c_y-r:c_y+r] \\
0 & , others
\end{aligned}
\right.},
\end{equation}
where $(c_x, c_y)$ is the center of the image and $r$ indicates the frequency threshold that distinguishes between high- and low-frequencies of the original image. Then, the low-pass filtered image ($LFI$) and high-pass filtered image ($HFI$) can be obtained as follows:
 %
\begin {align}
LFI &= F^{-}(F(I) \circ M),\\ 
HFI &= I - LFI,
\end {align}
%
where $\circ$ denotes the Hadamard product of the matrix.
And the obtained $LFI$ and $HFI$ are used as the CAE's optimization targets. For training the disentangler $D(\cdot)$ and reconstructor $R(\cdot)$ to correctly reconstruct the $HFI$ and $LFI$, the reconstructed loss of the CAE over multiple source domains is defined as
%
\begin {align}
\mathcal{L}_{cae} = \frac{1}{K} {\sum_{i=1}^{K}} \frac{1}{N_i} \sum_{j=1}^{N_i} {\Vert {X_{f_{j}}^i - \hat{X}_{f_{j}}^i}}\Vert_{2}^{2},
\end {align}
%
where $X_{f} \in \{LFI, HFI\}$, $\hat{X}_{f}$ is the output of $R(\cdot)$.

By using $HFI$ and $LFI$ as target labels for high-pass filtered image and low-pass filtered image reconstruction respectively, we can make the embedded features $f_H$ or $f_L$ more biased towards high-frequency features or low-frequency features of the image.

Meanwhile, we employ the $f_H$ and $f_L$ as input to train the auxiliary classifier ($C_{A_H}$, $C_{A_L}$) to correctly predict the sample class, which allows frequency-specific features to obtain corresponding semantic information.
%and speeds up the training of the network. 
This can be achieved by minimizing the standard cross-entropy loss:%
\begin{align}
    \mathcal{L}_{ca} = - \frac{1}{K} \sum_{i=1}^K \frac{1}{N_i} \sum_{j=1}^{N_i} {y_j^i} \log(C_A(AvgPool(f_{F_j}^i))),
\end{align}%
where $f_F \in \{f_H, f_L\}$, $C_A \in \{C_{A_H}, C_{A_L}\}$.

\subsection{Information Interaction Mechanism}
As analyzed in Sec.~\ref{sec1}, to make full use of the helpful information of both $f_H$ and $f_L$, we establish a practical information interaction mechanism. Inspired by \cite{lin2015,woo2018cbam}, we use the $f_L$ extracted in Sec.~\ref{sec3.1} to generate the corresponding spatial masks and multiply them with the $f_H$ to encode where to emphasize or suppress. 

To be specific, for features $f_L\in R^ {C\times H\times W}$ and $f_H \in R^ {C\times H\times W}$ extracted from the same sample, where $C, H, W$ respectively denote the number of channels, height, and width of the feature map, first, we use average-pooling and max-pooling for $f_L$ along the channel axis to obtain $f_{L_{avg}}\in R^ {1\times H\times W}$  and $f_{L_{max}}\in R^ {1\times H\times W}$, and then concatenate them to generate the feature $f_{spatial}\in R^ {2\times H\times W}$ that can effectively highlight spatial information. After that, the feature $f_{spatial}$ is fed into a standard convolutional layer to obtain 2D spatial mask $f_{mask}$. Formula is as follows:
%
\begin{align}
   f_{mask}=\sigma(Conv([AvgPool(f_L),MaxPool(f_L)])),
\end{align}
%
where $\sigma$ denotes sigmoid function and $Conv$ denotes a standard convolutional layer. Then we multiply the spatial mask $f_{mask}$ by $f_H$ to obtain the fused feature $f_Z$. 

After that, we use $f_Z$ as the input of classifier $C_I$ consisting of a fully connected layer to correctly predict the class of the each image, which is supervised by the cross-entropy loss:
%
\begin{align}
    \mathcal{L}_{ci} = -\frac{1}{K} \sum_{i=1}^K \frac{1}{N_i} \sum_{j=1}^{N_i} {y_j^i} \log(C_I(AvgPool(f_{Z_j}^{i}))).
\end{align}%

Intuitively, by jointly training $f_H$ and $f_L$, our method can trade off the ability of data-based feature representation, We hope to use the interaction between $f_H$ and $f_L$ for mutual constraints, 
which may allow the network to learn object edge features in high-frequency features while also noticing helpful information in low-frequency features. It is worth noting that our method is simple but has a significant effect on the generalization of the network, which proves the validity of the idea of interaction between high- and low-frequency features.

\begin{figure}[t]
    \centering
    \includegraphics[width=7cm]{ag_new.pdf}
    \caption{Frequency-domain-based data augmentation. Multiplicative and additive noises are both applied to the phase and amplitude of the image's frequency domain.}
    \label{fig:3}
\end{figure}

\subsection{Frequency-domain-based Data Augmentation}
The above method can make a large improvement in the generalization ability of the network, however, we cannot guarantee that it can also extract the high- and low-frequency features of the image well on the unseen domain, which will directly affect the robustness of the model. To alleviate this problem, intuitively, we propose a simple but effective data augmentation technique that works on the frequency domain.

First, we obtain the frequency domain representation of the image using Eq.~\ref{Eq.1}. And then we convert it to polar coordinate form:
%
\begin{align}
    F(u,v) = |F(u,v)| e^{-j\phi(u,v)},
\end{align}
%
then we can obtain the mathematical expressions for its amplitude and phase:
%
\begin{align}
    A(u,v) &= |F(u,v)|, \\
    P(u,v) &= \phi(u,v).
\end{align}%

In this paper, we apply frequency domain disturbance to enrich the diversity of sample distribution. Many previous work~\cite{xu2021,yang2020fda} argue the phase of the image has high-level semantic information, so they are both working on the amplitude such as exchanging the amplitude information between different images without changing the phase. We generalize this method by simply applying random noise to amplitude. In addition, we find that a slight disturbance to the phase can further improve the generalization ability of the network, and the experimental results are listed in Tab.~$\ref{tab:6}$. Specifically, inspired by \cite{li2021}, we utilize multiplicative and additive noise in the phase and amplitude. The computation formula is as follows:
%
\begin{align}
    \hat{A_g} = \alpha \circ A_g + \beta,
\end{align}%
where $A_g \in \{P, A\}$, 
%$\circ$ denotes the Hadamard product of the matrix, 
$\alpha \in R^{C\times H\times W}$ and $\beta \in R^{C\times H\times W}$ are multiplicative noise and additive noise, respectively. For example, each element $\alpha$ is sampled from Uniform distributions $U(a,b)$ and $\beta$ is sampled from Normal distributions $N(\mu,\sigma ^2)$. We then feed $\hat{A_g}$ into the inverse Fourier transform to obtain the augmented image. The frequency-domain-based data augmentation technique (FDAG) is shown in Fig.~\ref{fig:3}.

\subsection{Algorithm Flow}
The above three components together form the FFDI framework. In the training stage, we first augment the data using FDAG, then feed the data to the network
%: (i) obtain the embedded features $f_E$ of the input data by feature extractor $E(\cdot)$; (ii) utilize the CAE structure for feature disentanglement to obtain $f_H$ and $f_L$; (iii) the information interaction mechanism $IIM(\cdot)$ is applied for feature fusion $f_Z$; (iv) feed $f_Z$ to the classifier $C_I(\cdot)$ 
and use the overall loss $\mathcal{L}_{all}$ to train our model as follow:
%
\begin{align}
\label{Eq.15}
    \mathcal{L}_{all} = \mathcal{L}_{ci} + \lambda(\mathcal{L}_{ca_{L}} + \mathcal{L}_{ca_{H}} +  \mathcal{L}_{cae_L} + \mathcal{L}_{cae_H}).
\end{align}

After training, only $E(\cdot)$, $D(\cdot)$, $IIM(\cdot)$, and $C_I(\cdot)$ will be deployed for inference.
