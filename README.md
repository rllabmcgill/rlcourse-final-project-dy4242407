# rlcourse-final-project-dy4242407
rlcourse-final-project-dy4242407 created by GitHub Classroom


In this project, we implemented the TD($\lambda$) control method based on paper \cite{ryang2012framework} for the task of automatic text summarization. We implemented all code from scratch including the score function, Features encoding, Summarization environment, and the TD($\lambda$) control method. 

In addition, we implemented the policy gradient method, actor critic with eligibility trace, in the hope of developing a better algorithm for the task. Actor critic has been used successfully (faster learner than SARSA($\lambda$) from our experience in the environment of mountain car problem. However, it is not a better algorithm than TD($\lambda$) control method in the present setting of extractive sentence selection summarization. 
We believe this is due to the deterministic environment  and the large action set.
