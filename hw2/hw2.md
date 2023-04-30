# Reinforce Learning hw2

---

## Problem3 (a.)

---

In this problem, I only train one model for this problem.
With the default parameter in the provided file.

* Hidden size => 128
* Noise scale => 0.1
* Tau => 0.005
* Gamma => 0.995
* Batch size => 128

<img src="/Users/tutuchen/Desktop/RL/hw2/a_1.png" alt="a_1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/a_2.png" alt="a_2" style="zoom:70%;" />

As we can see, the model has siginificant improve after about 50 episodes whose policy can really finish our request.

---

## Problem3 (b.)

---

Using the same code as Problem (3a.) but do not have a model that can really finish the request perfectly every time.

The parameter I used will be present in the following way:

Episodes_Gamma_Tau_Hidden size_Noise scale

First I tried different Hidden size for this problem as I think it was more complex problem; therefore, the more hidden size may make it easier to find the fit policy for it.

<img src="/Users/tutuchen/Desktop/RL/hw2/b_hidden_size1.png" alt="b_hidden_size1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/b_hidden_size2.png" alt="b_hidden_size2" style="zoom:70%;" />

As we can see, the higher layer doesn't stand for well model on difficult problem.

I watched some of results render video, and I found that it may stuck on the air and didn't choose to land.

Thus, I thought this may be the problem that it didn't explore enough for this problem. This time, I chose to change noise scale value and hoped it will improve the exploration of the model.

<img src="/Users/tutuchen/Desktop/RL/hw2/b_noise_scale1.png" alt="b_noise_scale1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/b_noise_scale2.png" alt="b_noise_scale2" style="zoom:70%;" />

This time, it got the better score on the middle of our train. Nevertheless, it just much earlier found the no-landing status so I think it did not really help me to find better model.

After that, I attempted to different $$\gamma$$ to make our model be more concerned about the current action rewards.

<img src="/Users/tutuchen/Desktop/RL/hw2/gamma_1.png" alt="gamma_1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/gamma_2.png" alt="gamma_2" style="zoom:70%;" />

The result told me that it found a better way to fly lower on the no-landing status! Maybe it was not a bad news for me.

Consequently, I made the same test as I made for the $$\gamma = 0.95$$ model.
I tried to change the noise scale to make it has better exploration capability.

<img src="/Users/tutuchen/Desktop/RL/hw2/b_0.9_1.png" alt="b_0.9_1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/b_0.9_2.png" alt="b_0.9_2" style="zoom:70%;" />

Every model just find the no-landing state and kept on it.

I thought maybe it will be helpful to change $$\tau$$  and made our model has more update on the correct direction when it found how to land.

<img src="/Users/tutuchen/Desktop/RL/hw2/b_0.9_tau1.png" alt="b_0.9_tau1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/b_0.9_tau2.png" alt="b_0.9_tau2" style="zoom:70%;" />

It had better perform than the $$\tau = 0.005$$ case on beginning, whereas it did not have outstanding or stead improvement after that.

After finishing all this experiment, I try to enhance the episode value to make my model has more time to learn anything about landing.

<img src="/Users/tutuchen/Desktop/RL/hw2/2000_1.png" alt="2000_1" style="zoom:70%;" />

<img src="/Users/tutuchen/Desktop/RL/hw2/2000_2.png" alt="2000_2" style="zoom:70%;" />

The times it success is indeed increasing as the training time getting longer. I also change its noise scale value on this experiment. I have a conclusion is that higher noise scale value truely help the model to explore on the early time; nonetheless, it may make the model has worse performance when it has to exploitation as the better exploration makes its searching space be larger. 