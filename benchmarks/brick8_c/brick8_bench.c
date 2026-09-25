/*
 * brick8_bench.c -- faire Laufzeitmessung brick8 analytisch vs. KI in C.
 *
 * Beide Elementroutinen sind im gleichen Stil implementiert (einfache
 * Schleifen, kein BLAS, clang -O3), Ke wird in beiden Faellen nur als
 * oberes Dreieck aufsummiert und dann gespiegelt.
 *
 *   analytisch : element_brick8_nl.m (Total Lagrange, StVenant 3D,
 *                2x2x2 Gauss, B-Matrix-Form, dichte 6x6-Materialmatrix)
 *   KI         : element_brick8_nl_ai.m (Kanonisierung, modale Metrik-
 *                Kette, GELU-MLP mit Wert, Gradient und projizierter
 *                Hessian aus Vorwaerts-Tangenten, Rueckskalierung) inkl.
 *                OOD-Proxy ||E_green|| im Zentrum.
 *
 * Aufruf:
 *   brick8_bench <netz.bin> <referenz.bin> [wiederholungen]
 *
 * netz.bin      : export_c_weights.py (aus brick8_nl_W_network*.mat)
 * referenz.bin  : export_c_reference.m (MATLAB: Eingaben + Ke/Finte beider
 *                 Elemente fuer Zufallselemente und Benchmark-Struktur 5)
 *
 * Ausgabe: max. rel. Abweichung C vs. MATLAB fuer beide Routinen und
 * Mikrosekunden pro Elementaufruf.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef USE_BLAS
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#define NN 8
#define ND 24
#define NK 7
#define NM 28
#define NC 24
#define NIN (NC + NM)
#define MAXL 8
#define MAXH 512

/* ------------------------------------------------------------------ */
/* Netz                                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    int L;
    int rows[MAXL], cols[MAXL];
    double *W[MAXL], *b[MAXL];
    double c_mean[NC], c_std[NC], D_scale[NM];
} Net;

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static void read_or_die(void *p, size_t sz, size_t n, FILE *f) {
    if (fread(p, sz, n, f) != n) die("Lesefehler");
}

static Net load_net(const char *path) {
    Net N;
    FILE *f = fopen(path, "rb");
    if (!f) die("Netzdatei nicht lesbar");
    int32_t magic, L;
    read_or_die(&magic, 4, 1, f);
    if (magic != 0x44464C45) die("falsches Netzformat");
    read_or_die(&L, 4, 1, f);
    if (L > MAXL) die("zu viele Lagen");
    N.L = L;
    for (int l = 0; l < L; l++) {
        int32_t r, c;
        read_or_die(&r, 4, 1, f);
        read_or_die(&c, 4, 1, f);
        if (r > MAXH || c > MAXH) die("Lage zu breit");
        N.rows[l] = r;
        N.cols[l] = c;
        N.W[l] = malloc(sizeof(double) * r * c);
        N.b[l] = malloc(sizeof(double) * r);
        read_or_die(N.W[l], 8, (size_t)r * c, f);
        read_or_die(N.b[l], 8, r, f);
    }
    if (N.cols[0] != NIN || N.rows[L - 1] != 1) die("Netzgroesse passt nicht");
    read_or_die(N.c_mean, 8, NC, f);
    read_or_die(N.c_std, 8, NC, f);
    read_or_die(N.D_scale, 8, NM, f);
    fclose(f);
    return N;
}

/* ------------------------------------------------------------------ */
/* Gemeinsame Konstanten                                                */
/* ------------------------------------------------------------------ */
static const double RN[NN] = {-1, 1, 1, -1, -1, 1, 1, -1};
static const double SN[NN] = {-1, -1, 1, 1, -1, -1, 1, 1};
static const double TN[NN] = {-1, -1, -1, -1, 1, 1, 1, 1};
static double GP[8][3];
static double PHI[NN][NK];           /* [r s t | rs st rt rst] / 8 */
static int TI[NM], TJ[NM];            /* triu column-major */

static void init_const(void) {
    double a = 1.0 / sqrt(3.0);
    double g[8][3] = {{-a, -a, -a}, {a, -a, -a}, {a, a, -a}, {-a, a, -a},
                      {-a, -a, a},  {a, -a, a},  {a, a, a},  {-a, a, a}};
    memcpy(GP, g, sizeof(g));
    for (int i = 0; i < NN; i++) {
        double r = RN[i], s = SN[i], t = TN[i];
        double v[NK] = {r, s, t, r * s, s * t, r * t, r * s * t};
        for (int k = 0; k < NK; k++) PHI[i][k] = v[k] / 8.0;
    }
    int q = 0;
    for (int j = 0; j < NK; j++)
        for (int i = 0; i <= j; i++) { TI[q] = i; TJ[q] = j; q++; }
}

static int inv3(const double A[3][3], double B[3][3]) {
    double d = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
             - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
             + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    double id = 1.0 / d;
    B[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * id;
    B[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id;
    B[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id;
    B[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * id;
    B[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id;
    B[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id;
    B[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * id;
    B[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id;
    B[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id;
    return d > 0;
}

/* dh/dx und detJ an Punkt (r,s,t) */
static double shape(const double X[NN][3], double r, double s, double t, double dh[NN][3]) {
    double dr[NN][3], J[3][3] = {{0}}, iJ[3][3];
    for (int a = 0; a < NN; a++) {
        double ar = 1 + RN[a] * r, as = 1 + SN[a] * s, at = 1 + TN[a] * t;
        dr[a][0] = RN[a] * as * at / 8.0;
        dr[a][1] = ar * SN[a] * at / 8.0;
        dr[a][2] = ar * as * TN[a] / 8.0;
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 3; k++) J[i][k] += X[a][i] * dr[a][k];
    }
    double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
               - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
               + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    inv3(J, iJ);
    for (int a = 0; a < NN; a++)
        for (int j = 0; j < 3; j++)
            dh[a][j] = dr[a][0] * iJ[0][j] + dr[a][1] * iJ[1][j] + dr[a][2] * iJ[2][j];
    return det;
}

/* ------------------------------------------------------------------ */
/* Analytisches Element (StVenant 3D)                                   */
/* ------------------------------------------------------------------ */
static void element_ana(const double X[NN][3], const double *Ue, double E, double nu,
                        double *Ke, double *Fi) {
    double lam = E * nu / ((1 + nu) * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
    double C[6][6] = {{0}};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) C[i][j] = lam;
        C[i][i] += 2 * mu;
        C[i + 3][i + 3] = mu;
    }
    memset(Ke, 0, sizeof(double) * ND * ND);
    memset(Fi, 0, sizeof(double) * ND);
    for (int g = 0; g < 8; g++) {
        double dh[NN][3];
        double dV = shape(X, GP[g][0], GP[g][1], GP[g][2], dh);
        double F[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int a = 0; a < NN; a++)
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < 3; j++) F[k][j] += Ue[3 * a + k] * dh[a][j];
        double Eg[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double s = 0;
                for (int k = 0; k < 3; k++) s += F[k][i] * F[k][j];
                Eg[i][j] = 0.5 * (s - (i == j));
            }
        double Ev[6] = {Eg[0][0], Eg[1][1], Eg[2][2], 2 * Eg[0][1], 2 * Eg[1][2], 2 * Eg[0][2]};
        double Sv[6];
        for (int i = 0; i < 6; i++) {
            double s = 0;
            for (int j = 0; j < 6; j++) s += C[i][j] * Ev[j];
            Sv[i] = s;
        }
        double S[3][3] = {{Sv[0], Sv[3], Sv[5]}, {Sv[3], Sv[1], Sv[4]}, {Sv[5], Sv[4], Sv[2]}};
        double B[6][ND];
        for (int a = 0; a < NN; a++) {
            double hx = dh[a][0], hy = dh[a][1], hz = dh[a][2];
            for (int k = 0; k < 3; k++) {
                int c = 3 * a + k;
                B[0][c] = F[k][0] * hx;
                B[1][c] = F[k][1] * hy;
                B[2][c] = F[k][2] * hz;
                B[3][c] = F[k][0] * hy + F[k][1] * hx;
                B[4][c] = F[k][1] * hz + F[k][2] * hy;
                B[5][c] = F[k][0] * hz + F[k][2] * hx;
            }
        }
        for (int c = 0; c < ND; c++) {
            double s = 0;
            for (int i = 0; i < 6; i++) s += B[i][c] * Sv[i];
            Fi[c] += s * dV;
        }
        double CB[6][ND];
        for (int i = 0; i < 6; i++)
            for (int c = 0; c < ND; c++) {
                double s = 0;
                for (int j = 0; j < 6; j++) s += C[i][j] * B[j][c];
                CB[i][c] = s * dV;
            }
        for (int r = 0; r < ND; r++)
            for (int c = r; c < ND; c++) {
                double s = 0;
                for (int i = 0; i < 6; i++) s += B[i][r] * CB[i][c];
                Ke[r * ND + c] += s;
            }
        for (int a = 0; a < NN; a++) {
            double t[3];
            for (int j = 0; j < 3; j++)
                t[j] = dh[a][0] * S[0][j] + dh[a][1] * S[1][j] + dh[a][2] * S[2][j];
            for (int b = a; b < NN; b++) {
                double G = (t[0] * dh[b][0] + t[1] * dh[b][1] + t[2] * dh[b][2]) * dV;
                for (int k = 0; k < 3; k++) Ke[(3 * a + k) * ND + 3 * b + k] += G;
            }
        }
    }
    for (int r = 0; r < ND; r++)
        for (int c = 0; c < r; c++) Ke[r * ND + c] = Ke[c * ND + r];
}

/* ------------------------------------------------------------------ */
/* KI-Element (modale Metrik-Kette + Energienetz)                        */
/* ------------------------------------------------------------------ */
static volatile double g_ood;
static double Aw[MAXL][MAXH], D1w[MAXL][MAXH], D2w[MAXL][MAXH];
static double Zd[MAXL][MAXH * ND], Tmp[MAXH * ND];

/* Vorwaerts + Gradient; bei T != NULL zusaetzlich T'HT (ND x ND) */
static double mlp(const Net *N, const double *x, double *g, const double *T, double *THT) {
    int L = N->L;
    const double *a = x;
    double fval = 0;
    for (int l = 0; l < L; l++) {
        int R = N->rows[l], Cc = N->cols[l];
        const double *W = N->W[l], *b = N->b[l];
        if (l == L - 1) {
            double s = b[0];
            for (int j = 0; j < Cc; j++) s += W[j] * a[j];
            fval = s;
            break;
        }
        for (int i = 0; i < R; i++) {
            const double *w = W + (size_t)i * Cc;
            double z = b[i];
            for (int j = 0; j < Cc; j++) z += w[j] * a[j];
            double Phi = 0.5 * (1 + erf(z * M_SQRT1_2));
            double phi = exp(-0.5 * z * z) * 0.3989422804014327;
            Aw[l][i] = z * Phi;
            D1w[l][i] = Phi + z * phi;
            D2w[l][i] = (2 - z * z) * phi;
        }
        if (T) {                          /* Vorwaerts-Tangenten Zd[l] (R x ND) */
            double *Z = Zd[l];
            if (l == 0) {                 /* nur D-Eingaenge tragen Tangenten */
                for (int i = 0; i < R; i++) {
                    const double *w = W + (size_t)i * Cc + NC;
                    for (int d = 0; d < ND; d++) {
                        double s = 0;
                        for (int q = 0; q < NM; q++) s += w[q] * T[q * ND + d];
                        Z[i * ND + d] = s;
                    }
                }
            } else {
                const double *Zp = Zd[l - 1];
                int Rp = N->rows[l - 1];
                for (int j = 0; j < Rp; j++)
                    for (int d = 0; d < ND; d++) Tmp[j * ND + d] = D1w[l - 1][j] * Zp[j * ND + d];
                for (int i = 0; i < R; i++) {
                    const double *w = W + (size_t)i * Cc;
                    double *zi = Z + i * ND;
                    for (int d = 0; d < ND; d++) zi[d] = 0;
                    for (int j = 0; j < Cc; j++) {
                        double wij = w[j];
                        const double *tj = Tmp + j * ND;
                        for (int d = 0; d < ND; d++) zi[d] += wij * tj[d];
                    }
                }
            }
        }
        a = Aw[l];
    }
    /* Reverse, r = df/da_l */
    static double r[MAXH], rn[MAXH];
    int Rl = N->rows[L - 2];
    for (int i = 0; i < Rl; i++) r[i] = N->W[L - 1][i];
    if (T) memset(THT, 0, sizeof(double) * ND * ND);
    for (int l = L - 2; l >= 0; l--) {
        int R = N->rows[l], Cc = N->cols[l];
        if (T) {
            const double *Z = Zd[l];
            for (int i = 0; i < R; i++) {
                double wi = D2w[l][i] * r[i];
                const double *zi = Z + i * ND;
                for (int p = 0; p < ND; p++) {
                    double s = wi * zi[p];
                    for (int q = p; q < ND; q++) THT[p * ND + q] += s * zi[q];
                }
            }
        }
        for (int j = 0; j < Cc; j++) rn[j] = 0;
        const double *W = N->W[l];
        for (int i = 0; i < R; i++) {
            double ti = D1w[l][i] * r[i];
            const double *w = W + (size_t)i * Cc;
            for (int j = 0; j < Cc; j++) rn[j] += w[j] * ti;
        }
        memcpy(r, rn, sizeof(double) * Cc);
    }
    memcpy(g, r, sizeof(double) * N->cols[0]);
    return fval;
}

#ifdef USE_BLAS
/* Variante mit Accelerate-BLAS: gleiche Mathematik, Matrixprodukte als dgemm/dgemv */
static double Yw[MAXH * ND];
static double mlp_blas(const Net *N, const double *x, double *g, const double *T, double *THT) {
    int L = N->L;
    const double *a = x;
    double fval = 0;
    static double z[MAXH];
    for (int l = 0; l < L; l++) {
        int R = N->rows[l], Cc = N->cols[l];
        const double *W = N->W[l], *b = N->b[l];
        if (l == L - 1) {
            fval = b[0] + cblas_ddot(Cc, W, 1, a, 1);
            break;
        }
        memcpy(z, b, sizeof(double) * R);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, R, Cc, 1.0, W, Cc, a, 1, 1.0, z, 1);
        for (int i = 0; i < R; i++) {
            double zi = z[i];
            double Phi = 0.5 * (1 + erf(zi * M_SQRT1_2));
            double phi = exp(-0.5 * zi * zi) * 0.3989422804014327;
            Aw[l][i] = zi * Phi;
            D1w[l][i] = Phi + zi * phi;
            D2w[l][i] = (2 - zi * zi) * phi;
        }
        if (T) {
            if (l == 0) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R, ND, NM, 1.0,
                            W + NC, Cc, T, ND, 0.0, Zd[0], ND);
            } else {
                int Rp = N->rows[l - 1];
                const double *Zp = Zd[l - 1];
                for (int j = 0; j < Rp; j++)
                    for (int d = 0; d < ND; d++) Tmp[j * ND + d] = D1w[l - 1][j] * Zp[j * ND + d];
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R, ND, Cc, 1.0,
                            W, Cc, Tmp, ND, 0.0, Zd[l], ND);
            }
        }
        a = Aw[l];
    }
    static double r[MAXH], t[MAXH];
    int Rl = N->rows[L - 2];
    memcpy(r, N->W[L - 1], sizeof(double) * Rl);
    if (T) memset(THT, 0, sizeof(double) * ND * ND);
    for (int l = L - 2; l >= 0; l--) {
        int R = N->rows[l], Cc = N->cols[l];
        if (T) {
            const double *Z = Zd[l];
            for (int i = 0; i < R; i++) {
                double wi = D2w[l][i] * r[i];
                for (int d = 0; d < ND; d++) Yw[i * ND + d] = wi * Z[i * ND + d];
            }
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ND, ND, R, 1.0,
                        Z, ND, Yw, ND, 1.0, THT, ND);
        }
        for (int i = 0; i < R; i++) t[i] = D1w[l][i] * r[i];
        cblas_dgemv(CblasRowMajor, CblasTrans, R, Cc, 1.0, N->W[l], Cc, t, 1, 0.0, r, 1);
    }
    memcpy(g, r, sizeof(double) * N->cols[0]);
    return fval;
}
#define MLP mlp_blas
#else
#define MLP mlp
#endif

static double element_ai(const Net *N, const double X[NN][3], const double *Ue, double E,
                         double *Ke, double *Fi) {
    /* 1. Kanonisierung */
    double cen[3] = {0, 0, 0}, xb[NN][3], Lc = 0;
    for (int a = 0; a < NN; a++)
        for (int i = 0; i < 3; i++) cen[i] += X[a][i] / NN;
    for (int a = 0; a < NN; a++) {
        double s = 0;
        for (int i = 0; i < 3; i++) { xb[a][i] = X[a][i] - cen[i]; s += xb[a][i] * xb[a][i]; }
        Lc += sqrt(s) / NN;
    }
    for (int a = 0; a < NN; a++)
        for (int i = 0; i < 3; i++) xb[a][i] /= Lc;
    double e1[3], e2[3], e3[3], av[3], n1 = 0, n2 = 0, dp = 0;
    for (int i = 0; i < 3; i++) { e1[i] = xb[1][i] - xb[0][i]; n1 += e1[i] * e1[i]; }
    n1 = sqrt(n1);
    for (int i = 0; i < 3; i++) { e1[i] /= n1; av[i] = xb[3][i] - xb[0][i]; dp += av[i] * e1[i]; }
    for (int i = 0; i < 3; i++) { e2[i] = av[i] - dp * e1[i]; n2 += e2[i] * e2[i]; }
    n2 = sqrt(n2);
    for (int i = 0; i < 3; i++) e2[i] /= n2;
    e3[0] = e1[1] * e2[2] - e1[2] * e2[1];
    e3[1] = e1[2] * e2[0] - e1[0] * e2[2];
    e3[2] = e1[0] * e2[1] - e1[1] * e2[0];
    double ch[NN][3];
    for (int a = 0; a < NN; a++) {
        ch[a][0] = xb[a][0] * e1[0] + xb[a][1] * e1[1] + xb[a][2] * e1[2];
        ch[a][1] = xb[a][0] * e2[0] + xb[a][1] * e2[1] + xb[a][2] * e2[2];
        ch[a][2] = xb[a][0] * e3[0] + xb[a][1] * e3[1] + xb[a][2] * e3[2];
    }
    /* 2. modale Basis L = [dh0 | gamma] aus kanonischer Geometrie */
    double J0[3][3] = {{0}}, iJ0[3][3], Lm[NN][NK], Ah[3][4] = {{0}};
    for (int a = 0; a < NN; a++)
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) J0[i][k] += ch[a][i] * PHI[a][k];
            for (int h = 0; h < 4; h++) Ah[i][h] += ch[a][i] * PHI[a][3 + h];
        }
    inv3(J0, iJ0);
    for (int a = 0; a < NN; a++) {
        for (int j = 0; j < 3; j++)
            Lm[a][j] = PHI[a][0] * iJ0[0][j] + PHI[a][1] * iJ0[1][j] + PHI[a][2] * iJ0[2][j];
        for (int h = 0; h < 4; h++)
            Lm[a][3 + h] = PHI[a][3 + h] - (Lm[a][0] * Ah[0][h] + Lm[a][1] * Ah[1][h] + Lm[a][2] * Ah[2][h]);
    }
    /* 3. Zustand Z0 = L'xbar, Uz = L'uhat, D */
    double Z0[NK][3] = {{0}}, Uz[NK][3] = {{0}}, Z[NK][3];
    for (int a = 0; a < NN; a++)
        for (int k = 0; k < NK; k++)
            for (int d = 0; d < 3; d++) {
                Z0[k][d] += Lm[a][k] * xb[a][d];
                Uz[k][d] += Lm[a][k] * Ue[3 * a + d] / Lc;
            }
    for (int k = 0; k < NK; k++)
        for (int d = 0; d < 3; d++) Z[k][d] = Z0[k][d] + Uz[k][d];
    double x[NIN], x0[NIN], Dv[NM];
    for (int a = 0; a < NN; a++)
        for (int d = 0; d < 3; d++) {
            int i = 3 * a + d;
            x[i] = x0[i] = (ch[a][d] - N->c_mean[i]) / N->c_std[i];
        }
    for (int q = 0; q < NM; q++) {
        int i = TI[q], j = TJ[q];
        double s = 0;
        for (int d = 0; d < 3; d++) s += Z0[i][d] * Uz[j][d] + Uz[i][d] * Z0[j][d] + Uz[i][d] * Uz[j][d];
        Dv[q] = s;
        x[NC + q] = s / N->D_scale[q];
        x0[NC + q] = 0;
    }
    /* 4. Jf = dD/du_hat (NM x ND), skaliert mit 1/Ds */
    static double Jfs[NM * ND];
    for (int q = 0; q < NM; q++) {
        int i = TI[q], j = TJ[q];
        double is = 1.0 / N->D_scale[q];
        for (int a = 0; a < NN; a++)
            for (int d = 0; d < 3; d++)
                Jfs[q * ND + 3 * a + d] = (Z[i][d] * Lm[a][j] + Z[j][d] * Lm[a][i]) * is;
    }
    /* 5. Netz: echter Zustand (mit Hessian) und Nullzustand */
    static double g1[NIN], g0[NIN], THT[ND * ND];
    double f1 = MLP(N, x, g1, Jfs, THT);
    double f0 = MLP(N, x0, g0, NULL, NULL);
    double What = f1 - f0, p[NM];
    for (int q = 0; q < NM; q++) {
        What -= g0[NC + q] * x[NC + q];
        p[q] = (g1[NC + q] - g0[NC + q]) / N->D_scale[q];
    }
    /* 6. Kette zurueck */
    double S[NK][NK];
    for (int q = 0; q < NM; q++) {
        int i = TI[q], j = TJ[q];
        S[i][j] = p[q];
        S[j][i] = p[q];
        if (i == j) S[i][i] = 2 * p[q];
    }
    double SZ[NK][3] = {{0}}, LS[NN][NK] = {{0}};
    for (int i = 0; i < NK; i++)
        for (int j = 0; j < NK; j++)
            for (int d = 0; d < 3; d++) SZ[i][d] += S[i][j] * Z[j][d];
    for (int a = 0; a < NN; a++)
        for (int j = 0; j < NK; j++)
            for (int i = 0; i < NK; i++) LS[a][j] += Lm[a][i] * S[i][j];
    double sF = E * Lc * Lc, sK = E * Lc;
    for (int a = 0; a < NN; a++)
        for (int d = 0; d < 3; d++) {
            double s = 0;
            for (int k = 0; k < NK; k++) s += Lm[a][k] * SZ[k][d];
            Fi[3 * a + d] = sF * s;
        }
    for (int r = 0; r < ND; r++)
        for (int c = r; c < ND; c++) Ke[r * ND + c] = sK * THT[r * ND + c];
    for (int a = 0; a < NN; a++)
        for (int b = a; b < NN; b++) {
            double G = 0;
            for (int j = 0; j < NK; j++) G += LS[a][j] * Lm[b][j];
            for (int d = 0; d < 3; d++) Ke[(3 * a + d) * ND + 3 * b + d] += sK * G;
        }
    for (int r = 0; r < ND; r++)
        for (int c = 0; c < r; c++) Ke[r * ND + c] = Ke[c * ND + r];
    /* 7. OOD-Proxy: ||E_green|| im Zentrum (wie im MATLAB-Element) */
    double dh[NN][3], F[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, en = 0;
    shape(X, 0, 0, 0, dh);
    for (int a = 0; a < NN; a++)
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++) F[k][j] += Ue[3 * a + k] * dh[a][j];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += F[k][i] * F[k][j];
            s = 0.5 * (s - (i == j));
            en += s * s;
        }
    g_ood = sqrt(en);                 /* volatile: wird nicht wegoptimiert */
    return E * Lc * Lc * Lc * What;
}

/* ------------------------------------------------------------------ */
static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static double relerr(const double *a, const double *b, int n) {
    double d = 0, s = 0;
    for (int i = 0; i < n; i++) { d += (a[i] - b[i]) * (a[i] - b[i]); s += b[i] * b[i]; }
    return sqrt(d / (s > 0 ? s : 1));
}

int main(int argc, char **argv) {
    if (argc < 3) die("Aufruf: brick8_bench <netz.bin> <referenz.bin> [wiederholungen]");
    int reps = argc > 3 ? atoi(argv[3]) : 200;
    init_const();
    Net N = load_net(argv[1]);

    FILE *f = fopen(argv[2], "rb");
    if (!f) die("Referenzdatei nicht lesbar");
    int32_t nset;
    read_or_die(&nset, 4, 1, f);
#ifdef USE_BLAS
    const char *variant = "MLP mit Accelerate-BLAS";
#else
    const char *variant = "MLP mit einfachen Schleifen";
#endif
    printf("Netz: %s | Lagen %d | Breite %d | %s\n", argv[1], N.L, N.rows[0], variant);
    for (int s = 0; s < nset; s++) {
        int32_t ne, hasAI;
        double E, nu;
        read_or_die(&ne, 4, 1, f);
        read_or_die(&hasAI, 4, 1, f);
        read_or_die(&E, 8, 1, f);
        read_or_die(&nu, 8, 1, f);
        double *X = malloc(sizeof(double) * ne * 24), *U = malloc(sizeof(double) * ne * 24);
        double *Ka = malloc(sizeof(double) * ne * 576), *Fa = malloc(sizeof(double) * ne * 24);
        double *Kk = malloc(sizeof(double) * ne * 576), *Fk = malloc(sizeof(double) * ne * 24);
        read_or_die(X, 8, (size_t)ne * 24, f);
        read_or_die(U, 8, (size_t)ne * 24, f);
        read_or_die(Ka, 8, (size_t)ne * 576, f);
        read_or_die(Fa, 8, (size_t)ne * 24, f);
        read_or_die(Kk, 8, (size_t)ne * 576, f);
        read_or_die(Fk, 8, (size_t)ne * 24, f);

        double eKa = 0, eFa = 0, eKk = 0, eFk = 0, Ke[576], Fi[24];
        for (int e = 0; e < ne; e++) {
            const double (*Xe)[3] = (const double (*)[3])(X + 24 * e);
            element_ana(Xe, U + 24 * e, E, nu, Ke, Fi);
            double v;
            v = relerr(Ke, Ka + 576 * e, 576); if (v > eKa) eKa = v;
            v = relerr(Fi, Fa + 24 * e, 24);   if (v > eFa) eFa = v;
            if (hasAI) {
                element_ai(&N, Xe, U + 24 * e, E, Ke, Fi);
                v = relerr(Ke, Kk + 576 * e, 576); if (v > eKk) eKk = v;
                v = relerr(Fi, Fk + 24 * e, 24);   if (v > eFk) eFk = v;
            }
        }
        /* Zeitmessung: alle Elemente des Satzes, reps-mal, Median aus 5 Laeufen */
        double tA[5], tK[5], chk = 0;
        for (int run = 0; run < 5; run++) {
            double t0 = now_s();
            for (int r = 0; r < reps; r++)
                for (int e = 0; e < ne; e++) {
                    element_ana((const double (*)[3])(X + 24 * e), U + 24 * e, E, nu, Ke, Fi);
                    chk += Ke[r % 576];
                }
            tA[run] = (now_s() - t0) / ((double)reps * ne);
            t0 = now_s();
            for (int r = 0; r < reps; r++)
                for (int e = 0; e < ne; e++) {
                    chk += element_ai(&N, (const double (*)[3])(X + 24 * e), U + 24 * e, E, Ke, Fi);
                    chk += Ke[r % 576];
                }
            tK[run] = (now_s() - t0) / ((double)reps * ne);
        }
        for (int i = 0; i < 5; i++)          /* Median */
            for (int j = i + 1; j < 5; j++) {
                if (tA[j] < tA[i]) { double t = tA[i]; tA[i] = tA[j]; tA[j] = t; }
                if (tK[j] < tK[i]) { double t = tK[i]; tK[i] = tK[j]; tK[j] = t; }
            }
        printf("Satz %d: %d Elemente | C vs MATLAB: analyt. Ke %.1e Fi %.1e", s + 1, ne, eKa, eFa);
        if (hasAI) printf(" | KI Ke %.1e Fi %.1e", eKk, eFk);
        printf("\n        Zeit/Element: analytisch %.2f us | KI %.2f us | Verhaeltnis KI/analyt. %.2f   (chk %.3e)\n",
               1e6 * tA[2], 1e6 * tK[2], tK[2] / tA[2], chk);
        free(X); free(U); free(Ka); free(Fa); free(Kk); free(Fk);
    }
    fclose(f);
    return 0;
}
