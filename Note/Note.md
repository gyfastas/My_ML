# Note

## Supervised Learning

### Multivariate Linear Regression

The gradient descent for multiple feature:
$$
\theta_j :=\theta_j -\alpha \frac 1m \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)}).x_j^{(i)}
$$
----------------



**feature scaling ** and **mean normalization**:

To make the feature in range -1 to 1:
$$
X_n = \frac{X_n-u}{S_n}
$$
where $u$ is the average of $X_n$, $S_n$ is the standard deviation.

----------------



**learning rate**: 

if sufficient small , $j(\theta)$ will decrease on every iteration







****



### Polynomial Regression

If we use polynomial regression, feature scaling is important!

#### Normal Equation

$$
h_\theta(x)=\theta^TX \\
goal: y = \theta^TX\\
X^TX\theta = X^Ty\\
multiply(X^TX)^{-1}\\
\theta  = (X^TX)^{-1}X^Ty
$$

slow if n is very large

$X^TX$ should be invertible! If no...

### Logistic Regression

This is for classification!!!

#### Sigmoid function

$$
h_\theta(x) = g(\theta^Tx)\\
g(z) = \frac1{1+e^{-z}}
$$

$h_\theta(x)$ = estimated P that y = 1 on input x

#### Decision boundary

For non-linear decision boundaries: Polynomial Re

#### Convex cost function

redefine $J(\theta)$ in log :

#### Advanced Optimization



#### Multi-class Classification

One-vs-all : 构建多个二分类器

pick the class i that max $h_\theta(x)$





### Decision Tree

#### Basic Concept

Mutual information (from wiki): https://en.wikipedia.org/wiki/Mutual_information

**Entropy:**
$$
Ent(D) = -\sum_{k=1}^{|y|}p_k\log_2p_k
$$
​	最好的情况: $p_k$ =1 (此时Ent(D)为0)

**Information Gain**



**Gain ratio**:

信息增益具有自己的弱点 <例如"编号"作为划分>---偏好取值较多的属性！

def Gain ratio:
$$
Gain_ratio(D,a) = \frac{Gain(D,a)}{IV(a)} \\
$$


#### ID3 and C4.5









### SVM





## Semi-Supervised Learning

## Unsupervised Learning



# Project and Resources







