import numpy as np

class tinympcref:
    def __init__(self):
        self.nx = 0 # number of states
        self.nu = 0 # number of control inputs
        self.N = 0 # number of knotpoints in the horizon
        self.A = [] # state transition matrix
        self.B = [] # control matrix
        self.Q = [] # state cost matrix (diagonal)
        self.R = [] # input cost matrix (digaonal)
        self.rho = 0
        self.x_min = [] # lower bounds on state
        self.x_max = [] # upper bounds on state
        self.u_min = [] # lower bounds on input
        self.u_max = [] # upper bounds on input

        self.iter = 0
        self.solved = 0
        
        self.primal_residual_state = 0.0
        self.dual_residual_state = 0.0
        self.primal_residual_input = 0.0
        self.dual_residual_input = 0.0
        
    def setup(self, A, B, Q, R, N, rho=1.0,
              x_min=None, x_max=None, u_min=None, u_max=None, verbose=False, **settings):
        self.rho = rho
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        assert Q.shape[0] == Q.shape[1]
        assert A.shape[0] == Q.shape[0]
        assert R.shape[0] == R.shape[1]
        assert B.shape[1] == R.shape[0]

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.nx = A.shape[0]
        self.nu = B.shape[1]

        assert N > 1
        self.N = N

        self.x = np.zeros((self.N, self.nx))
        self.u = np.zeros((self.N-1, self.nu))

        self.q = np.zeros((self.N, self.nx))
        self.r = np.zeros((self.N-1, self.nu))

        self.p = np.zeros((self.N, self.nx))
        self.d = np.zeros((self.N-1, self.nu))

        self.v = np.zeros((self.N, self.nx))
        self.vnew = np.zeros((self.N, self.nx))
        self.z = np.zeros((self.N-1, self.nu))
        self.znew = np.zeros((self.N-1, self.nu))

        self.g = np.zeros((self.N, self.nx))
        self.y = np.zeros((self.N-1, self.nu))

        self.Q = (self.Q + self.rho * np.eye(self.nx)).diagonal()
        self.R = (self.R + self.rho * np.eye(self.nu)).diagonal()

        if x_min is not None:
            self.x_min = np.tile(x_min, (self.N, 1))
        else:
            self.x_min = -np.inf * np.ones((self.N, self.nx))

        if x_max is not None:
            self.x_max = np.tile(x_max, (self.N, 1))
        else:
            self.x_max = np.inf * np.ones((self.N, self.nx))
            
        if u_min is not None:
            self.u_min = np.tile(u_min, (self.N-1, 1))
        else:
            self.u_min = -np.inf * np.ones((self.N-1, self.nu))

        if u_max is not None:
            self.u_max = np.tile(u_max, (self.N-1, 1))
        else:
            self.u_max = np.inf * np.ones((self.N-1, self.nu))

        self.Xref = np.zeros((self.N, self.nx))
        self.Uref = np.zeros((self.N-1, self.nu))

        self.Qu = np.zeros(self.nu)

        Q1 = np.diag(self.Q) + rho * np.eye(self.nx)
        R1 = np.diag(self.R) + rho * np.eye(self.nu)

        if(verbose):
            print('A = ', A)
            print('B = ', B)
            print('Q = ', self.Q)
            print('R = ', self.R)
            print('rho = ', rho)

        Ktp1 = np.zeros((self.nu, self.nx))
        Ptp1 = rho * np.eye(self.nx)
        Kinf = np.zeros((self.nu, self.nx))
        Pinf = np.zeros((self.nx, self.nx))

        for i in range(1000):
            Kinf = np.linalg.inv(R1 + B.T @ Ptp1 @ B) @ (B.T @ Ptp1 @ A)
            Pinf = Q1 + A.T @ Ptp1 @ (A - B @ Kinf)

            if np.max(np.abs(Kinf - Ktp1)) < 1e-5:
                if(verbose):
                    print('Kinf converged after %d iterations' % i)
                break

            Ktp1 = Kinf
            Ptp1 = Pinf

        Quu_inv = np.linalg.inv(R1 + B.T @ Pinf @ B)
        AmBKt = (A - B @ Kinf).T

        self.Kinf = Kinf
        self.Pinf = Pinf
        self.Quu_inv = Quu_inv
        self.AmBKt = AmBKt

        if(verbose):
            print('Kinf:', Kinf)
            print('Pinf:', Pinf)
            print('Quu_inv:', Quu_inv)
            print('AmBKt:', AmBKt)
        
        if 'abs_pri_tol' in settings:
            self.abs_pri_tol = settings.pop('abs_pri_tol')
        else:
            self.abs_pri_tol = 1e-3
        if 'abs_dua_tol' in settings:
            self.abs_dua_tol = settings.pop('abs_dua_tol')
        else:
            self.abs_dua_tol = 1e-3
        if 'max_iter' in settings:
            self.max_iter = settings.pop('max_iter')
        else:
            self.max_iter = 100
        if 'check_termination' in settings:
            self.check_termination = settings.pop('check_termination')
        else:
            self.check_termination = 25
        if 'en_state_bound' in settings:
            self.en_state_bound = 1 if settings.pop('en_state_bound') else 0
        if 'en_input_bound' in settings:
            self.en_input_bound = 1 if settings.pop('en_input_bound') else 0
        
    def set_x0(self, x0):
        assert len(x0) == self.nx
        self.x[0] = x0

    def set_x_ref(self, x_ref):
        self.Xref = x_ref
    
    def set_u_ref(self, u_ref):
        self.Uref = u_ref

    def _forward_pass(self):
        for i in range(self.N-1):
            self.u[i] = -self.Kinf @ self.x[i] - self.d[i]
            self.x[i+1] = self.A @ self.x[i] + self.B @ self.u[i]

    def _update_slack(self):
        self.znew = self.u + self.y
        self.vnew = self.x + self.g

        # box constraints on input
        self.znew = np.minimum(self.u_max, np.maximum(self.u_min, self.znew))

        self.vnew = np.minimum(self.x_max, np.maximum(self.x_min, self.vnew))
    
    def _update_dual(self):
        self.y = self.y + self.u - self.znew
        self.g = self.g + self.x - self.vnew

    def _update_linear_cost(self):
        for i in range(self.N-1):
            self.r[i] = -self.Uref[i] * self.R

        for i in range(self.N):
            self.q[i] = -self.Xref[i] * self.Q

        self.r -= self.rho * (self.znew - self.y)
        self.q -= self.rho * (self.vnew - self.g)

        # self.p[self.N-1] = -self.Xref[self.N-1].T @ self.Pinf
        self.p[self.N-1] = -self.Pinf.T @ self.Xref[self.N-1]

        # print('Xref:', self.Xref)
        self.p[self.N-1] -= self.rho * (self.vnew[self.N-1] - self.g[self.N-1])

    def _backward_pass(self):
        for i in range(self.N-2, -1, -1):
            self.d[i] = self.Quu_inv @ (self.B.T @ self.p[i+1] + self.r[i])
            self.p[i] = self.q[i] + self.AmBKt @ self.p[i+1] - self.Kinf.T @ self.r[i]

    def _check_termination(self):
        if self.check_termination == 0:
            return False

        if self.iter % self.check_termination == 0:
            self.primal_residual_state = np.max(np.abs(self.x - self.vnew))
            self.dual_residual_state = np.max(np.abs(self.v - self.vnew)) * self.rho
            self.primal_residual_input = np.max(np.abs(self.u - self.znew))
            self.dual_residual_input = np.max(np.abs(self.z - self.znew)) * self.rho

            if (self.primal_residual_state < self.abs_pri_tol and
                self.primal_residual_input < self.abs_pri_tol and
                self.dual_residual_state < self.abs_dua_tol and
                self.dual_residual_input < self.abs_dua_tol):
                return True
        return False

    def solve(self):
        self.iter = 0
        self.solved = 0

        for i in range(self.max_iter):
            self._forward_pass()
            self._update_slack()

            self._update_dual()
            self._update_linear_cost()

            self.iter += 1
            if self._check_termination():
                self.solved = 1
                break
            
            self.v = self.vnew
            self.z = self.znew

            self._backward_pass()
        
        self.x = self.vnew
        self.u = self.znew 