# EMBGO
Efficient multiplayer battle game optimizer for numerical optimization and adversarial robust neural architecture search

## Highlights
• We quantitatively analyze the deficiency of the primary multiplayer battle game optimizer (MBGO).  
• We propose an efficient MBGO (EMBGO) by introducing novel differential mutation and Lévy flight.  
• Comprehensive numerical experiments on CEC2017, CEC2020, CEC2022, and eight engineering problems confirm the competitiveness of EMBGO.  
• We further apply the proposed EMBGO to solve adversarial robust neural architecture search (ARNAS) tasks.  

## Abstract
This paper introduces a novel metaheuristic algorithm, known as the efficient multiplayer battle game optimizer (EMBGO), specifically designed for addressing complex numerical optimization tasks. The motivation behind this research stems from the need to rectify identified shortcomings in the original MBGO, particularly in search operators during the movement phase, as revealed through ablation experiments. EMBGO mitigates these limitations by integrating the movement and battle phases to simplify the original optimization framework and improve search efficiency. Besides, two efficient search operators: differential mutation and Lévy flight are introduced to increase the diversity of the population. To evaluate the performance of EMBGO comprehensively and fairly, numerical experiments are conducted on benchmark functions such as CEC2017, CEC2020, and CEC2022, as well as engineering problems. Twelve well-established MA approaches serve as competitor algorithms for comparison. Furthermore, we apply the proposed EMBGO to the complex adversarial robust neural architecture search (ARNAS) tasks and explore its robustness and scalability. The experimental results and statistical analyses confirm the efficiency and effectiveness of EMBGO across various optimization tasks. As a potential optimization technique, EMBGO holds promise for diverse applications in real-world problems and deep learning scenarios. The source code of EMBGO is made available in https://github.com/RuiZhong961230/EMBGO.


## Citation
@article{Zhong:25,  
title = {Efficient multiplayer battle game optimizer for numerical optimization and adversarial robust neural architecture search},  
journal = {Alexandria Engineering Journal},  
volume = {113},  
pages = {150-168},  
year = {2025},  
issn = {1110-0168},  
doi = {https://doi.org/10.1016/j.aej.2024.11.035 },  
author = {Rui Zhong and Yuefeng Xu and Chao Zhang and Jun Yu},  
}

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library, engineering problems are provided by the enoppy library, and the dataset of adversarial robustness neural architecture search (ARNAS) is downloaded from https://steffen-jung.github.io/robustness/.
