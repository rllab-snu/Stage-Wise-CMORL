# Definition of Reward and Cost Functions

## Go1 Back-Flip

|Stage|Stand|Sit|Jump|Air|Land|Threshold|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | |**Reward Functions**| | | |
|Base Height| $-\|p_z - 0.35\|$ | $-\|p_z - 0.2\|$ | $\mathbf{1}_{(p_z \leq 0.5)}\cdot p_z$ | $\mathbf{1}_{(p_z \leq 0.5)}\cdot p_z$ | $-\|p_z - 0.35\|$ | |
|Base Velocity| $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-\mathbf{1}_\mathrm{turn}\cdot \omega_y$ | $-\mathbf{1}_\mathrm{turn}\cdot \omega_y$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | |
|Base Balance| $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\|\angle (\hat{y}_B, \hat{z}_W)-\pi/2\|$ | $-\|\angle (\hat{y}_B, \hat{z}_W)-\pi/2\|$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | |
|Energy| $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | |
|Style| $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | |
| | | |**Cost Functions**| | | |
|Foot Contact|-|-| $\mathbf{1}_{(\|I_C^\mathrm{foot,rear}\| = 0)}$ |-|-| $0.25$ |
|Body Contact| $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $0.025$ |
|Joint Position| $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $0.025$ |
|Joint Velocity| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $0.025$ |
|Joint Torque| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $0.025$ |


## Go1 Side-Flip

|Stage|Stand|Sit|Jump|Air|Land|Threshold|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | |**Reward Functions**| | | |
|Base Height| $-\|p_z - 0.35\|$ | $-\|p_z - 0.2\|$ | $\mathbf{1}_{(p_z \leq 0.5)}\cdot p_z$ | $\mathbf{1}_{(p_z \leq 0.5)}\cdot p_z$ | $-\|p_z - 0.35\|$ | |
|Base Velocity| $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | $\mathbf{1}_\mathrm{turn}\cdot \omega_x$ | $\mathbf{1}_\mathrm{turn}\cdot \omega_x$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | |
|Base Balance| $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\|\angle (\hat{x}_B, \hat{z}_W)-\pi/2\|$ | $-\|\angle (\hat{x}_B, \hat{z}_W)-\pi/2\|$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | |
|Energy| $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | |
|Style| $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | |
| | | |**Cost Functions**| | | |
|Foot Contact|-|-| $\mathbf{1}_{(\|I_C^\mathrm{foot,right}\| = 0)}$ |-|-| $0.25$ |
|Body Contact| $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $0.025$ |
|Joint Position| $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $0.025$ |
|Joint Velocity| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $0.025$ |
|Joint Torque| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $0.025$ |

## Go1 Side-Roll

|Stage|Stand|Sit|Roll-Half|Roll-Full|Recover|Threshold|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | |**Reward Functions**| | | |
|Base Height| $-\|p_z - 0.35\|$ | $-\|p_z - 0.1\|$ | $-\mathbf{1}_{(p_z \geq 0.2)}\cdot p_z$ | $-\mathbf{1}_{(p_z \geq 0.2)}\cdot p_z$ | $-\|p_z - 0.35\|$ | |
|Base Balance| $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\|\angle (\hat{x}_B, \hat{z}_W)-\pi/2\|$ | $-\|\angle (\hat{x}_B, \hat{z}_W)-\pi/2\|$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | |
|Rotation Angle| $0$ | $0$ | $-\angle (-\hat{z}_B, \hat{z}_W)$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | $0$ | |
|Base Velocity| $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | $\mathbf{1}_{(\omega_x\leq 4\pi)}\cdot \omega_x$ | $\mathbf{1}_{(\omega_x\leq 4\pi)}\cdot \omega_x$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | |
|Energy| $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | |
|Style| $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{down}_j)^2$ | $-\sum_j (q_j-q^\mathrm{down}_j)^2$ | $-\sum_j (q_j-q^\mathrm{down}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | |
| | | |**Cost Functions**| | | |
|Foot Contact|-|$1 - \|I_C^\mathrm{foot}\|/4$|-|-|-| $0.25$ |
|Body Contact| $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ |-|-|-|-| $0.025$ |
|Joint Position| $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $0.025$ |
|Joint Velocity| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $0.025$ |
|Joint Torque| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $0.025$ |

## Go1 Two-Hand Walk

|Stage|Stand|Tilt|Walk|Threshold|
|:---:|:---:|:---:|:---:|:---:|
| | |**Reward Functions**| | |
|Base Balance| $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\|\angle (\hat{z}_B, \hat{z}_W)-\pi/4\|$ | $-\|\angle (\hat{z}_B, \hat{z}_W)-\pi/4\|$ | |
|Base Velocity| $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-((v_x-v_x^\mathrm{cmd})^2 + (v_y-v_y^\mathrm{cmd})^2 + (\omega_z - \omega_z^\mathrm{cmd})^2)$ | |
|Energy| $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | |
|Style| $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | |
|$1\mathrm{st}$ Smoothness| $-\sum_j (a_{j,t} - a_{j,t-1})^2$ | $-\sum_j (a_{j,t} - a_{j,t-1})^2$ | $-\sum_j (a_{j,t} - a_{j,t-1})^2$ | |
|$2\mathrm{nd}$ Smoothness| $-\sum_j (a_{j,t} - 2a_{j,t-1} + a_{j,t-2})^2$ | $-\sum_j (a_{j,t} - 2a_{j,t-1} + a_{j,t-2})^2$ | $-\sum_j (a_{j,t} - 2a_{j,t-1} + a_{j,t-2})^2$ | |
| | |**Cost Functions**| | |
|Foot Contact|-|-|$(1-\|I_C^\mathrm{foot, FL}\|)\cdot(1-\|I_C^\mathrm{foot, FR}\|) + \|I_C^\mathrm{foot, rear}\|/2$|$0.025$|
|Base Height|$\mathbf{1}_{(p_z < 0.3)}$|$\mathbf{1}_{(p_z < 0.3)}$|$\mathbf{1}_{(p_z < 0.3)}$| $0.025$ |
|Body Contact| $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $0.025$ |
|Joint Position| $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $0.025$ |
|Joint Velocity| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $0.025$ |
|Joint Torque| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $0.025$ |

## H1 Back-Flip

|Stage|Stand|Sit|Jump|Air|Land|Threshold|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | |**Reward Functions**| | | |
|Base Height| $-\|p_z - 0.95\|$ | $-\|p_z - 0.6\|$ | $\mathbf{1}_{(p_z \leq 1.8)}\cdot p_z$ | $\mathbf{1}_{(p_z \leq 1.8)}\cdot p_z$ | $-\|p_z - 0.95\|$ | |
|Base Velocity| $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-\mathbf{1}_\mathrm{turn}\cdot \omega_y$ | $-\mathbf{1}_\mathrm{turn}\cdot \omega_y$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | |
|Base Balance| $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\|\angle (\hat{y}_B, \hat{z}_W)-\pi/2\|$ | $-\|\angle (\hat{y}_B, \hat{z}_W)-\pi/2\|$ | $-\angle (\hat{z}_B, \hat{z}_W)$ | |
|Energy| $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | |
|Style| $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{sit}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | |
| | | |**Cost Functions**| | | |
|Foot Contact| $\mathbf{1}_{(\|I_C^\mathrm{foot}\| = 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{foot}\| = 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{foot}\| = 0)}$ |-|-| $0.25$ |
|Body Contact| $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $0.025$ |
|Joint Position| $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $0.025$ |
|Joint Velocity| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $0.025$ |
|Joint Torque| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $0.025$ |

## H1 Two-Hand Walk

|Stage|Stand|Tilt|Walk|Threshold|
|:---:|:---:|:---:|:---:|:---:|
| | |**Reward Functions**| | |
|Base Balance| $-\angle (\hat{z}_B, \hat{z}_W)$ | $-\|\angle (\hat{z}_B, \hat{z}_W)-\pi/2\|$ | $-\|\angle (\hat{z}_B, \hat{z}_W)-\pi\|$ | |
|Base Velocity| $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | $-(v_x^2 + v_y^2 + \omega_z^2)$ | |
|Contact Penalty| $1 - \|I_C^\mathrm{hand}\|/2$ | $(\|I_C^\mathrm{hand}\| + \|I_C^\mathrm{foot}\|)/2$ | $1 - \|I_C^\mathrm{foot}\|/2$ | |
|Energy| $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | $-\sum_j \tau_j^2$ | |
|Style| $-\sum_j (q_j-q^\mathrm{default}_j)^2$ | $-\sum_j (q_j-q^\mathrm{down}_j)^2$ | $-\sum_j (q_j-q^\mathrm{reverse}_j)^2$ | |
|$1\mathrm{st}$ Smoothness| $-\sum_j (a_{j,t} - a_{j,t-1})^2$ | $-\sum_j (a_{j,t} - a_{j,t-1})^2$ | $-\sum_j (a_{j,t} - a_{j,t-1})^2$ | |
|$2\mathrm{nd}$ Smoothness| $-\sum_j (a_{j,t} - 2a_{j,t-1} + a_{j,t-2})^2$ | $-\sum_j (a_{j,t} - 2a_{j,t-1} + a_{j,t-2})^2$ | $-\sum_j (a_{j,t} - 2a_{j,t-1} + a_{j,t-2})^2$ | |
| | |**Cost Functions**| | |
|Foot Contact|$\mathbf{1}_{(\|I_C^\mathrm{foot}\| = 0)}$|-|$\mathbf{1}_{(\|I_C^\mathrm{hand}\| = 0)}$|$0.25$|
|Base Height|$\mathbf{1}_{(p_z < 1.25)}$|$\mathbf{1}_{(p_z < 0.5)}$|$\mathbf{1}_{(p_z < 0.5)}$| $0.025$ |
|Body Contact| $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $\mathbf{1}_{(\|I_C^\mathrm{body}\| > 0)}$ | $0.025$ |
|Joint Position| $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(q_j > q_j^\mathrm{max}\|\|q_j < q_j^\mathrm{min})}$ | $0.025$ |
|Joint Velocity| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\dot{q}_j \| > \dot{q}_j^\mathrm{max})}$ | $0.025$ |
|Joint Torque| $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $\frac{1}{J} \sum_j \mathbf{1}_{(\|\tau_j \| > \tau_j^\mathrm{max})}$ | $0.025$ |
