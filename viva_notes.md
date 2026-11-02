# Numerical Analysis Programming (NAP) Course: Comprehensive Viva Notes
### Smart Academic Planner & CGPA Predictor

This guide prepares you for university viva examinations by explaining each numerical method in clear, simple terms, its mathematical foundation, its edge cases, and its exact practical role in this academic planner.

---

## 1. Bisection Method

### Simple Explanation
Imagine playing a guessing game where a number is between 0 and 10. You test the middle (5). If the answer is "higher", you throw away [0, 5] and now search in [5, 10]. By repeatedly cutting the interval in half, you are guaranteed to trap the root in an increasingly tiny window.

### Mathematical Formula
$$c_k = \frac{a_k + b_k}{2}$$
If $f(a_k) \cdot f(c_k) < 0$, root lies in $[a_k, c_k] \implies b_{k+1} = c_k$.  
Else $a_{k+1} = c_k$.

### Theoretical Properties
- **Convergence Rate:** Linear ($r = 1$), error halves at each iteration: $|e_{k+1}| = \frac{1}{2} |e_k|$.
- **Complexity:** $O(\log_2((b - a) / \text{tol}))$.
- **Guarantee:** Unconditionally guaranteed to converge if $f(x)$ is continuous and $f(a) \cdot f(b) < 0$ (Bolzano's Intermediate Value Theorem).

### Project Application
- **Where Used:** `numerical/root_finding.py` & Target CGPA Planner page.
- **Problem Solved:** Given current credits $C_c$, current CGPA $G_c$, upcoming credits $C_u$, and target CGPA $T$:
  $$f(x) = \frac{C_c \cdot G_c + C_u \cdot x}{C_c + C_u} - T = 0$$
  Finds the exact upcoming SGPA $x \in [0, 10]$ needed.

### Key Viva Questions
- **Q: Can Bisection find roots of even multiplicity (e.g., $f(x) = (x-2)^2$)?**  
  *A:* No, because the function touches the axis without crossing it, so $f(a) \cdot f(b) \le 0$ cannot be bracketed.
- **Q: How does our code handle invalid inputs?**  
  *A:* It validates that $a < b$ and explicitly checks if $f(a) \cdot f(b) > 0$, raising a descriptive `ValueError`.

---

## 2. Newton-Raphson Method

### Simple Explanation
Instead of blindly halving intervals, draw a tangent line at your current guess. Follow the tangent line down to where it hits the x-axis, and use that intersection as your next guess. Because curves look like straight lines when zoomed in, this zooms in on the root with incredible speed.

### Mathematical Formula
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
Derived from first-order Taylor expansion: $f(x_{n+1}) \approx f(x_n) + f'(x_n)(x_{n+1} - x_n) = 0$.

### Theoretical Properties
- **Convergence Rate:** Quadratic ($r = 2$), meaning the number of correct decimal places doubles with each iteration near a simple root: $|e_{n+1}| \approx \frac{|f''(r)|}{2|f'(r)|} |e_n|^2$.
- **Complexity:** $O(k)$ evaluations (typically converges in 4–8 iterations).

### Project Application
- **Where Used:** `numerical/root_finding.py` & Mark Balancing module.
- **Problem Solved:** Computes the required final exam score under non-linear weightings (e.g. grading curves or diminishing return score mappings).

### Key Viva Questions
- **Q: What is the main failure mode of Newton-Raphson?**  
  *A:* When $f'(x_n) \approx 0$ (horizontal tangent), division by zero occurs or the step explodes to infinity. Our implementation detects $|f'(x_n)| < 10^{-14}$ and raises `ZeroDivisionError`.
- **Q: Compare Bisection vs. Newton-Raphson.**  
  *A:* Bisection is slow (linear) but 100% reliable if bracketed. Newton-Raphson is lightning fast (quadratic) but requires derivatives and can diverge if the initial guess $x_0$ is poor.

---

## 3. Fixed-Point Iteration

### Simple Explanation
Rewrite any equation $f(x) = 0$ into the self-referential form $x = g(x)$. Start with a guess $x_0$, compute $x_1 = g(x_0)$, then $x_2 = g(x_1)$, and keep feeding the output back into the input until the values stop changing.

### Mathematical Formula
$$x_{k+1} = g(x_k)$$
Convergence condition (Banach Fixed-Point Theorem): $|g'(x)| < 1$ in an interval around the fixed point $x^*$.

### Theoretical Properties
- **Convergence Rate:** Linear, with asymptotic error constant $C = |g'(x^*)|$.
- **Complexity:** $O(k)$ function evaluations.

### Project Application
- **Where Used:** `numerical/fixed_point.py` & Study Plan Generator.
- **Problem Solved:** Models diminishing returns of study hours vs. marks:
  $$M(x) = M_{\max} \cdot (1 - e^{-\alpha x})$$
  Rearranged with damping parameter into a contraction mapping $x = g(x)$ where $|g'(x)| < 1$ to calculate exact daily hours needed for an 85% target score.

### Key Viva Questions
- **Q: What happens if $|g'(x)| \ge 1$?**  
  *A:* The iterations spiral outward and diverge to infinity. Our implementation tracks step size and terminates with a descriptive warning if steps explode.

---

## 4. Gauss Elimination & Gauss-Jordan Elimination

### Simple Explanation
Given a system of equations (e.g., 3 equations with 3 unknown study hours), use row operations to eliminate variables one by one until you have a triangular staircase. In Gauss Elimination, you solve the bottom variable first and back-substitute upwards. In Gauss-Jordan, you clear above and below each pivot, transforming the matrix into the identity matrix so solutions appear directly.

### Mathematical Formulas
- **Forward Elimination:** $m_{ik} = \frac{A_{ik}}{A_{kk}}$, then $\text{Row}_i \leftarrow \text{Row}_i - m_{ik} \cdot \text{Row}_k$.
- **Back Substitution:** $x_i = \frac{b_i - \sum_{j=i+1}^n A_{ij} x_j}{A_{ii}}$.
- **Complexity:** Gauss is $O(\frac{n^3}{3})$ flops; Gauss-Jordan is $O(\frac{n^3}{2})$ flops.

### Project Application
- **Where Used:** `numerical/linear_systems.py` & Study Timetable Generator.
- **Problem Solved:** Allocates total available study hours $H$ across $n$ subjects subject to constraints:
  1. $\sum h_i = H$ (Total hours budget)
  2. $\frac{h_i}{w_i} - \frac{h_{i+1}}{w_{i+1}} = 0$ (Subject difficulty-credit parity)
  Also solves internal/midterm/end-term mark distribution systems.

### Key Viva Questions
- **Q: Why is Partial Pivoting mandatory?**  
  *A:* If a pivot $A_{kk} \approx 0$, dividing by it causes catastrophic floating-point overflow and destroys numerical accuracy. Partial pivoting swaps the current row with the row having the largest absolute value in that column $|\text{Row}_p|$, ensuring multipliers $|m_{ik}| \le 1$.
- **Q: How is a singular matrix detected?**  
  *A:* If the largest pivot candidate in a column is smaller than tolerance $\epsilon = 10^{-12}$, the matrix is singular or ill-conditioned, and our code raises `SingularMatrixError`.

---

## 5. Power Method

### Simple Explanation
Multiply any non-zero vector repeatedly by matrix $A$ and normalize it after each step. Because the dominant eigenvalue grows faster than all other eigenvalues, the vector naturally swings into alignment with the principal eigenvector, like a compass needle pointing North.

### Mathematical Formula
$$\mathbf{y}_k = A \mathbf{v}_{k-1}, \quad \mathbf{v}_k = \frac{\mathbf{y}_k}{\|\mathbf{y}_k\|_2}$$
Rayleigh Quotient for dominant eigenvalue:
$$\lambda_k = \frac{\mathbf{v}_k^T A \mathbf{v}_k}{\mathbf{v}_k^T \mathbf{v}_k}$$

### Theoretical Properties
- **Convergence Rate:** Ratio $\left|\frac{\lambda_2}{\lambda_1}\right|$. The smaller this ratio, the faster it converges.
- **Complexity:** $O(k \cdot n^2)$ for $k$ iterations on $n \times n$ matrix.

### Project Application
- **Where Used:** `numerical/power_method.py` & Study Plan Generator.
- **Problem Solved:** Constructs a positive pairwise priority matrix $A$ combining credits, weakness rating $(6 - \text{strength})$, pending syllabus %, and exam urgency. By the Perron-Frobenius theorem, the dominant eigenvector $\mathbf{v}_1$ yields the global study priority ranking for subjects.

### Key Viva Questions
- **Q: When will the Power Method fail to converge?**  
  *A:* If the matrix has two dominant eigenvalues of opposite sign ($\lambda_1 = -\lambda_2$), or if the initial vector $\mathbf{v}_0$ is strictly orthogonal to the dominant eigenvector.

---

## 6. Lagrange Interpolation

### Simple Explanation
To pass a smooth curve through $n+1$ points without solving a matrix, construct special "basis polynomials" $L_i(x)$ that equal $1$ at point $x_i$ and $0$ at all other data points. Then multiply each basis polynomial by its corresponding y-value and add them together.

### Mathematical Formula
$$P_n(x) = \sum_{i=0}^n y_i \cdot \ell_i(x), \quad \text{where } \ell_i(x) = \prod_{j \neq i}^n \frac{x - x_j}{x_i - x_j}$$

### Theoretical Properties
- **Complexity:** $O(n^2)$ to evaluate $P_n(x)$ at a point.
- **Degree:** Unique polynomial of degree $\le n$ passing through $n+1$ distinct points.

### Project Application
- **Where Used:** `numerical/lagrange.py` & CGPA Predictor page.
- **Problem Solved:** Estimates missing or future semester SGPA values based on completed historical semesters $(sem_i, SGPA_i)$.

### Key Viva Questions
- **Q: What is Runge's Phenomenon in polynomial interpolation?**  
  *A:* As the polynomial degree $n$ increases with equally spaced nodes, wild oscillations occur near the edges of the interval. This is why Hermite interpolation or splines are preferred for higher orders.

---

## 7. Hermite Interpolation

### Simple Explanation
Lagrange interpolation only cares about passing through points $(x_i, y_i)$. Hermite interpolation matches both the points **and their slopes** $(x_i, y_i, y'_i)$. This forces the curve to respect the trend (improvement velocity) of the student's grades and eliminates unrealistic overshoot.

### Mathematical Formula (Newton Confluent Divided Differences)
Double the nodes: $z_{2i} = z_{2i+1} = x_i$.  
$$Q[2i, 0] = y_i, \quad Q[2i+1, 0] = y_i, \quad Q[2i+1, 1] = y'_i$$
Higher orders: $Q[i, j] = \frac{Q[i, j-1] - Q[i-1, j-1]}{z_i - z_{i-j}}$.  
$$H_{2n+1}(x) = Q[0,0] + \sum_{k=1}^{2n+1} Q[k,k] \prod_{j=0}^{k-1} (x - z_j)$$

### Theoretical Properties
- **Degree:** $2n + 1$ (2 conditions per node).
- **Complexity:** $O(N^2)$ where $N = 2(n+1)$.

### Project Application
- **Where Used:** `numerical/hermite.py` & Numerical Methods Lab.
- **Problem Solved:** Uses semester SGPA plus semester-to-semester momentum $y'_i \approx \Delta y / \Delta x$ to predict the next semester SGPA, directly compared side-by-side with Lagrange.

---

## 8. Least Squares Regression

### Simple Explanation
When real-world data is noisy, forcing a high-degree polynomial to hit every single point is a mistake (overfitting). Instead, fit a straight line or quadratic curve that minimizes the sum of squared vertical distances (residuals) from the points to the line.

### Mathematical Formula
For $y = mx + c$, solve the Normal Equations:
$$\begin{bmatrix} N & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix} \begin{bmatrix} c \\ m \end{bmatrix} = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix}$$
Goodness of fit:
$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### Theoretical Properties
- **Complexity:** $O(N \cdot p^2 + p^3)$ where $p = \text{degree} + 1$.
- **Statistical Property:** Best Linear Unbiased Estimator (Gauss-Markov Theorem).

### Project Application
- **Where Used:** `numerical/least_squares.py` & Numerical Methods Lab.
- **Problem Solved:** Fits empirical curve to historical study hours vs. marks scored, providing $R^2$ accuracy and residual error tables.

---

## 9. Composite Trapezoidal Rule

### Simple Explanation
To find the area under an arbitrary curve, approximate the region as a series of adjacent vertical trapezoids. Calculate the area of each trapezoid ($\text{width} \times \frac{\text{left} + \text{right}}{2}$) and add them up.

### Mathematical Formula
$$\int_a^b f(x)dx \approx \frac{h}{2} \left[ f(a) + 2 \sum_{i=1}^{n-1} f(x_i) + f(b) \right], \quad h = \frac{b - a}{n}$$
Discrete: $\sum_{i=0}^{n-1} \frac{x_{i+1} - x_i}{2} (y_i + y_{i+1})$.

### Theoretical Properties
- **Global Error:** $E_T = -\frac{(b - a) h^2}{12} f''(\xi) = O(h^2)$.
- **Exact for:** Polynomials of degree $\le 1$ (linear functions).

### Project Application
- **Where Used:** `numerical/numerical_integration.py` & Attendance / Effort Tracker.
- **Problem Solved:** Integrates the continuous/discrete daily study hour distribution over a semester to calculate cumulative student study effort (hours $\times$ days).

---

## 10. Runge-Kutta Methods (RK4 & Adaptive RK45)

### Simple Explanation
Euler's method estimates the next point by looking only at the slope at the current point, which accumulates errors quickly. RK4 takes four clever slope samples across each step ($k_1$ at start, $k_2, k_3$ at midpoint, $k_4$ at end) and blends them with Simpson's 1/6 weighting. RK45 computes both a 4th and 5th order step simultaneously; if they disagree, it automatically shrinks the step size!

### Mathematical Formula (RK4)
$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + h/2, y_n + \frac{h}{2} k_1)$$
$$k_3 = f(t_n + h/2, y_n + \frac{h}{2} k_2)$$
$$k_4 = f(t_n + h, y_n + h k_3)$$
$$y_{n+1} = y_n + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$

### Theoretical Properties
- **RK4 Global Truncation Error:** $O(h^4)$.
- **RK45 (Fehlberg/Dormand-Prince):** Embedded pair with local truncation error estimate $\text{LTE} = |y_5 - y_4|$. Adaptively scales $h_{\text{new}} = h \cdot (\text{tol} / \text{LTE})^{0.2}$.

### Project Application
- **Where Used:** `numerical/runge_kutta.py` & Timetable Generator.
- **Problem Solved:** Simulates the Ebbinghaus forgetting curve ODE:
  $$\frac{dy}{dt} = k(1 - y) - d \cdot y$$
  where $y(t) \in [0, 1]$ is topic retention, $k$ is study rate, and $d$ is decay. Whenever retention drops below 70%, the study planner automatically triggers and schedules a **Revision Milestone Day**.
