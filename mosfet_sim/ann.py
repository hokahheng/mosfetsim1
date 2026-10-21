"""ANN hooks (optional, not used by the GUI yet).

NumpyMLP       tiny dependency-free regressor (fit / predict / save / load).
               Any object with fit(X, y) and predict(X) can replace it
               (scikit-learn MLPRegressor, a PyTorch wrapper, ...).
HybridModel    physics model + ANN residual.  It exposes the same interface as
               DeviceModel (id / gm / go / vth / any other attribute), so
               simulate(), the plots and the CSV export work with it unchanged.

    hyb = HybridModel(model)                       # starts identical to physics
    hyb.fit_residual(vgs, vds, id_measured)        # refine with measured data
    r = simulate(hyb, p)
"""
import numpy as np


class NumpyMLP:
    def __init__(self, hidden=(32, 32), lr=1e-2, epochs=2000, seed=0):
        self.hidden, self.lr, self.epochs, self.seed = tuple(hidden), lr, epochs, seed
        self.W = self.b = None

    def _forward(self, Xn):
        acts = [Xn]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))
        return acts, acts[-1] @ self.W[-1] + self.b[-1]

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(len(X), -1)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        self.ym, self.ys = y.mean(0), y.std(0) + 1e-12
        Xn, yn = (X - self.xm) / self.xs, (y - self.ym) / self.ys
        rng = np.random.default_rng(self.seed)
        sizes = [X.shape[1], *self.hidden, y.shape[1]]
        self.W = [rng.normal(0, 1 / np.sqrt(a), (a, c)) for a, c in zip(sizes, sizes[1:])]
        self.b = [np.zeros(c) for c in sizes[1:]]
        mW, vW = [np.zeros_like(w) for w in self.W], [np.zeros_like(w) for w in self.W]
        mb, vb = [np.zeros_like(b) for b in self.b], [np.zeros_like(b) for b in self.b]
        for t in range(1, self.epochs + 1):                      # full-batch Adam
            acts, out = self._forward(Xn)
            g = 2 * (out - yn) / len(Xn)
            for i in reversed(range(len(self.W))):
                gW, gb = acts[i].T @ g, g.sum(0)
                if i:
                    g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
                for P, G, M, V in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                    M[i] = 0.9 * M[i] + 0.1 * G
                    V[i] = 0.999 * V[i] + 0.001 * G ** 2
                    P[i] -= self.lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def predict(self, X):
        _, out = self._forward((np.asarray(X, float) - self.xm) / self.xs)
        return out * self.ys + self.ym

    def save(self, path):
        np.savez(path, xm=self.xm, xs=self.xs, ym=self.ym, ys=self.ys,
                 n=len(self.W), **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        z, net = np.load(path), cls()
        net.xm, net.xs, net.ym, net.ys = z["xm"], z["xs"], z["ym"], z["ys"]
        net.W = [z[f"W{i}"] for i in range(int(z["n"]))]
        net.b = [z[f"b{i}"] for i in range(int(z["n"]))]
        return net


class HybridModel:
    """Physics + learned residual:  Id = Id_physics + ANN(Vgs, Vds)."""

    def __init__(self, base, regressor=None):
        self.base = base
        self.reg = regressor or NumpyMLP(hidden=(24, 24), epochs=1500)
        self.fitted = False

    def __getattr__(self, name):                 # vth, label, width, p, ... -> physics
        return getattr(self.base, name)

    def fit_residual(self, vgs, vds, id_measured):
        vgs, vds, im = (np.asarray(a, float) for a in (vgs, vds, id_measured))
        self.reg.fit(np.column_stack([vgs, vds]), im - self.base.id(vgs, vds))
        self.fitted = True
        return self

    def id(self, vgs, vds):
        vgs, vds = np.broadcast_arrays(np.asarray(vgs, float), np.asarray(vds, float))
        i = self.base.id(vgs, vds)
        if not self.fitted:
            return i
        corr = self.reg.predict(np.column_stack([vgs.ravel(), vds.ravel()]))[:, 0]
        return i + corr.reshape(vgs.shape)

    def gm(self, vgs, vds, h=1e-4):              # derivatives by central difference
        return (self.id(np.asarray(vgs) + h, vds) - self.id(np.asarray(vgs) - h, vds)) / (2 * h)

    def go(self, vgs, vds, h=1e-4):
        vds = np.asarray(vds, float)
        lo = np.maximum(vds - h, 0)
        return (self.id(vgs, vds + h) - self.id(vgs, lo)) / (vds + h - lo)
