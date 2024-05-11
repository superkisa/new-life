# Transfer learning in locomotion problem via masked self-attention mechanism

This repository contains code of the master's thesis on the subject of 'Transfer learning in locomotion problem via masked self-attention mechanism'.

The main implementation code is located in `libs/bert_sac/` directory, in `libs/bert_sac/custom_hope.py` file.

## Картинки

<figure>
  <img
  src="assets/repo/images/legs_geom.jpg"
  alt="Ant legs geometry">
  <figcaption>Ant legs geometry.</figcaption>
</figure>

<figure>
  <img
  src="assets/repo/images/custom_model_structure.jpg"
  alt="Custom model structure">
  <figcaption>Custom model structure.</figcaption>
</figure>

<figure>
  <img
  src="assets/repo/images/bert_attention_scheme.jpg"
  alt="BERT attention scheme">
  <figcaption>BERT attention scheme.</figcaption>
</figure>

t - torso
fl - front left leg
fr - front right leg
bl - back left leg
br - back right leg

| ![space-1.jpg](http://www.storywarren.com/wp-content/uploads/2016/09/space-1.jpg) | 
|:--:| 
| *Space* |

## References

1. [MuJoCo's Ant-v4 (Gymnasium)](https://gymnasium.farama.org/environments/mujoco/ant/)
1. [CleanRL's SAC](https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy)
