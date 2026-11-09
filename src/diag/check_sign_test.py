import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
from scipy import stats

def main():
    print("=== Sign-test and Permutation Sequence Analysis ===")
    
    # 1. Permutation balance sign-test: 17 successes out of 30 trials with p=0.5
    # Two-sided binomial test (sign test) for whether 17 vs 13 is significantly different from 50/50:
    binom_res = stats.binomtest(k=17, n=30, p=0.5, alternative='two-sided')
    print(f"1. Permutation Balance (17 Row A=C5, 13 Row A=C0, N=30):")
    print(f"   Two-sided Binomial p-value: {binom_res.pvalue:.4f} ({binom_res.pvalue})")
    print(f"   Conclusion: p > 0.05, không có bằng chứng lệch tỷ lệ giữa Row A và Row B (hoàn toàn tương thích với hoán vị ngẫu nhiên 50/50).")
    
    # 2. Wald-Wolfowitz Runs Test for randomness of sequence with 8 switches (9 runs), n1=17, n2=13:
    n1 = 17
    n2 = 13
    N = n1 + n2
    R = 9 # 9 runs (8 switches)
    
    # Expected number of runs E(R) = 1 + 2*n1*n2 / N
    E_R = 1 + (2 * n1 * n2) / N
    # Variance Var(R) = 2*n1*n2 * (2*n1*n2 - N) / (N^2 * (N - 1))
    Var_R = (2 * n1 * n2 * (2 * n1 * n2 - N)) / (N**2 * (N - 1))
    std_R = Var_R ** 0.5
    
    # Z score with continuity correction
    z = (R - E_R) / std_R
    p_runs = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\n2. Wald-Wolfowitz Runs Test (Tính ngẫu nhiên chuỗi):")
    print(f"   Số khối (runs) R = {R} (8 lần đổi hàng/switches)")
    print(f"   Kỳ vọng E[R] = {E_R:.2f}, Độ lệch chuẩn sigma[R] = {std_R:.2f}")
    print(f"   Z-score = {z:.4f}")
    print(f"   Two-sided Runs test p-value: {p_runs:.4f} ({p_runs})")
    print(f"   (R = 9 thấp hơn E[R]=15.73 do hiệu ứng gom cụm theo lô 3-4 câu/lần đổi, p={p_runs:.4f} < 0.05).")

    # 3. Also check the Sign-Test of rater preference (C5 vs C0):
    # Overall 30q: 29 C5 wins vs 1 C0 win
    pref_binom_30 = stats.binomtest(k=29, n=30, p=0.5, alternative='two-sided')
    print(f"\n3. Rater Preference Sign-Test (C5 vs C0):")
    print(f"   Toàn bộ 30 câu (29 thắng, 1 thua): p-value = {pref_binom_30.pvalue:.4e}")
    pref_binom_26 = stats.binomtest(k=26, n=26, p=0.5, alternative='two-sided')
    print(f"   26 câu spec (26 thắng, 0 thua): p-value = {pref_binom_26.pvalue:.4e}")

if __name__ == "__main__":
    main()
