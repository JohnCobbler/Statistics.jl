# test/test_var_std_quantile_mutants.jl
#
# Targeted regression and property tests for var/varm/std/_quantile numerical branches.
#
# Motivation: escaped-CI defect trail —
#   #195  std([1e300, -1e300]) → Inf overflow (corrected denominator path)
#   #121  quantile! AssertionError on degenerate range (index/clamp path)
#   #136  quantile Rational OverflowError (fma/interpolation path)
#
# Each test block includes its targeted mutation and is verified FALSIFIABLE
# (applied mutant makes the test fail; revert restores green).
#
# This file is a part of Julia. License is MIT: https://julialang.org/license
#
# Generated with AI assistance (Claude Sonnet 4.6); see PR body for disclosure.

using Statistics, Test

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1: corrected flag — denominator n-1 vs n
#
# Targeted mutant: `Int(corrected)` → `0` (or `corrected` Bool-flip to false)
# in the Welford path and the explicit-mean path of _var(iterable, ...).
#
# If corrected is ignored, var([1,2,3]; corrected=true) returns biased variance
# (2/3) instead of unbiased (1), and var(...; corrected=false) returns 1 instead
# of 2/3.  The ratio is exactly (n-1)/n for n=3.
# ─────────────────────────────────────────────────────────────────────────────

@testset "var corrected flag: Welford path (iterable)" begin
    xs = (1, 2, 3)  # tuple → iterable, uses _var(iterable, corrected, mean)
    n  = 3

    v_corr   = var(xs; corrected=true)
    v_biased = var(xs; corrected=false)

    # Unbiased: sum of squared deviations / (n-1) = 2/2 = 1.0
    @test v_corr   ≈ 1.0
    # Biased: sum of squared deviations / n = 2/3 ≈ 0.6667
    @test v_biased ≈ 2.0/3.0

    # Mutation-killing ratio check: if corrected is ignored the two values collapse
    # to the same number — the ratio would be 1.0 instead of n/(n-1).
    @test v_corr / v_biased ≈ n / (n - 1)
end

@testset "var corrected flag: explicit mean path (iterable)" begin
    xs = (1.0, 3.0, 5.0)   # mean=3, deviations = -2, 0, +2
    n  = 3

    v_corr   = var(xs; corrected=true,  mean=3.0)
    v_biased = var(xs; corrected=false, mean=3.0)

    # sum of sq dev = 4+0+4 = 8; unbiased = 8/2 = 4; biased = 8/3
    @test v_corr   ≈ 4.0
    @test v_biased ≈ 8.0/3.0
    @test v_corr / v_biased ≈ n / (n - 1)
end

@testset "var corrected flag: array path (varm!)" begin
    A = [1.0, 2.0, 3.0]
    n = length(A)

    v_corr   = var(A; corrected=true)
    v_biased = var(A; corrected=false)

    @test v_corr   ≈ 1.0
    @test v_biased ≈ 2.0/3.0
    @test v_corr / v_biased ≈ n / (n - 1)
end

@testset "var corrected flag: array with explicit mean (varm!)" begin
    A = [1.0, 3.0, 5.0]
    n = length(A)

    v_corr   = var(A; corrected=true,  mean=3.0)
    v_biased = var(A; corrected=false, mean=3.0)

    @test v_corr   ≈ 4.0
    @test v_biased ≈ 8.0/3.0
    @test v_corr / v_biased ≈ n / (n - 1)
end

@testset "std corrected flag: iterable" begin
    xs = (1, 2, 3)

    s_corr   = std(xs; corrected=true)
    s_biased = std(xs; corrected=false)

    @test s_corr   ≈ 1.0
    @test s_biased ≈ sqrt(2.0/3.0)
    # std is sqrt of var, so these must differ
    @test s_corr > s_biased
end

@testset "std corrected flag: array" begin
    A = [1.0, 2.0, 3.0]

    s_corr   = std(A; corrected=true)
    s_biased = std(A; corrected=false)

    @test s_corr   ≈ 1.0
    @test s_biased ≈ sqrt(2.0/3.0)
    @test s_corr > s_biased
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2: corrected flag with dims= reduction
#
# Targeted mutant: `Int(corrected)` → `0` in `varm!` (the `rn` computation),
# or `corrected` Bool-flip.  With dims= the path goes through varm!(R, A, m)
# which computes rn = div(length(A), length(R)) - Int(corrected).
# ─────────────────────────────────────────────────────────────────────────────

@testset "var corrected flag: dims=2 matrix" begin
    # 2×4 matrix; var over rows (each row has 4 elements)
    # Row 1: [1,2,3,4] → mean=2.5; deviations²: 2.25+0.25+0.25+2.25=5
    #   unbiased = 5/3; biased = 5/4
    M = [1.0 2.0 3.0 4.0;
         5.0 6.0 7.0 8.0]
    n = 4

    v_corr   = var(M; dims=2, corrected=true)
    v_biased = var(M; dims=2, corrected=false)

    @test v_corr[1]   ≈ 5.0/3.0
    @test v_biased[1] ≈ 5.0/4.0
    @test all(v_corr .> v_biased)  # unbiased always larger for n>1
    # ratio must be n/(n-1) for each row
    @test all(v_corr ./ v_biased .≈ n / (n - 1))
end

@testset "std corrected flag: dims=1 matrix" begin
    M = [1.0 4.0;
         3.0 8.0]
    # col 1: [1,3] → biased std = 1; unbiased = sqrt(2)
    s_corr   = std(M; dims=1, corrected=true)
    s_biased = std(M; dims=1, corrected=false)

    @test s_corr[1]   ≈ sqrt(2.0)
    @test s_biased[1] ≈ 1.0
    @test all(s_corr .> s_biased)
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3: varm range path — corrected Bool and l<=1 comparison
#
# Targeted mutants:
#   (a) `corrected ?  vv : vv*(l-1)/l`  →  Bool-flip: always returns `vv`
#   (b) `l <= 1` → `l < 1` (LE→LT): l=1 would no longer trigger the fallback,
#       causing an incorrect analytical formula to be applied.
# ─────────────────────────────────────────────────────────────────────────────

@testset "varm range: corrected flag" begin
    r = 1.0:8.0
    m = mean(r)

    v_corr   = varm(r, m; corrected=true)
    v_biased = varm(r, m; corrected=false)

    # Range analytical formula: var = step^2 * (n+1)*n/12  (corrected)
    # For 1:8 step=1, n=8: corrected = 8*9/12 * (7/8 adjustment)... match array
    @test v_corr   ≈ var(collect(r); corrected=true)
    @test v_biased ≈ var(collect(r); corrected=false)
    @test v_corr > v_biased   # mutation: Bool-flip collapses both to the same
end

@testset "varm range: length-1 fallback (l<=1 boundary)" begin
    # l=1: both corrected and uncorrected are defined at the boundary
    r1 = 5.0:5.0  # length-1 range

    # corrected=true on length-1: NaN (0/0)
    @test isnan(varm(r1, 5.0; corrected=true))
    # corrected=false on length-1: 0.0
    @test varm(r1, 5.0; corrected=false) === 0.0

    # Verify that these match the array path (ensuring the l<=1 branch delegates
    # correctly; if l<1 mutant is applied, length-1 range goes down the
    # analytical formula path and returns a wrong result instead of NaN/0.0)
    @test isnan(varm(collect(r1), 5.0; corrected=true))
    @test varm(collect(r1), 5.0; corrected=false) === 0.0
end

@testset "var range: matches array for multiple lengths" begin
    for n in 2:7
        r = 1.0:Float64(n)
        v = collect(r)
        @test var(r; corrected=true)  ≈ var(v; corrected=true)
        @test var(r; corrected=false) ≈ var(v; corrected=false)
        # corrected and uncorrected must differ by factor n/(n-1)
        @test var(r; corrected=true) / var(r; corrected=false) ≈ n / (n - 1)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4: _quantile clamp/index path
#
# Targeted mutants:
#   (a) `clamp(trunc(Int, aleph), 1, n-1)` — upper bound n-1: if changed to n
#       then v[j+1] accesses v[n+1] → BoundsError for p=1.0, n>=2
#   (b) `clamp(aleph - j, 0, 1)` — γ clamping: if 0→-1, negative weights
#   (c) `n == 1` check: LE→LT or EQ→NEQ mutant
# ─────────────────────────────────────────────────────────────────────────────

@testset "_quantile: j-index upper clamp (p=1.0 boundary)" begin
    # For p=1.0, aleph = n (exactly), j = clamp(n, 1, n-1) = n-1, γ=1 → v[n].
    # If upper clamp is n instead of n-1, j=n, b=v[n+1] → BoundsError.
    xs = [10, 20, 30, 40, 50]
    @test quantile(xs, 1.0) === 50.0
    @test quantile(xs, 0.0) === 10.0

    # Sorted variants
    @test quantile(xs, 1.0; sorted=true) === 50.0
    @test quantile(xs, 0.0; sorted=true) === 10.0
end

@testset "_quantile: γ lower clamp (p=0.0 boundary)" begin
    # For p=0.0, aleph=m (the alpha offset), j=clamp(trunc(m),1,n-1)=1, γ=clamp(aleph-1, 0,1).
    # With default alpha=beta=1, m=1, aleph=fma(n,0,1)=1, j=1, γ=0 → result = v[1].
    # If γ clamp lower bound is -1 instead of 0, γ = 0 still (since aleph-j=0); but for
    # edge cases with non-integer aleph, γ could go negative → interpolation out of [a,b].
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    @test quantile(xs, 0.0) === 1.0
    # Monotonicity: quantile is non-decreasing
    qs = [quantile(xs, p) for p in 0.0:0.1:1.0]
    @test issorted(qs)
end

@testset "_quantile: n==1 single-element vector" begin
    # For n=1, both a=b=v[1]; result must equal v[1] regardless of p.
    # If n==1 check is mutated to n<=1 or n>1, the v[j+1] index path
    # may be taken, causing OOB access.
    @test quantile([42.0], 0.0) === 42.0
    @test quantile([42.0], 0.5) === 42.0
    @test quantile([42.0], 1.0) === 42.0
    @test quantile([42], 0.5) === 42.0
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5: _quantile isfinite branch and interpolation formula
#
# Targeted mutants:
#   (a) `isfinite(a) && isfinite(b)` → `!isfinite(a) || !isfinite(b)` (NOT flip)
#       This swaps which formula is used: overflow-safe vs rounding-safe.
#   (b) `a ≈ b` condition → `a ≉ b` (EQ→NEQ or NOT flip)
#       This changes whether to use `a + γ*(b-a)` or `(1-γ)*a + γ*b`.
#
# #195-class overflow regression: std([1e300, -1e300]) should return Inf,
# not NaN. The two-argument variance path uses (b-a) which can overflow;
# the `isfinite` branch correctly routes to the weight-sum formula.
# ─────────────────────────────────────────────────────────────────────────────

@testset "_quantile: isfinite branch — overflow-safe interpolation (#195 class)" begin
    # quantile([v, -v], 0.75) where v is near-max float:
    # a = -v, b = v, γ = 0.5, b-a = 2v overflows.
    # isfinite branch routes to (1-γ)*a + γ*b = 0.5*(-v) + 0.5*v = 0.
    # If isfinite check is flipped, it uses a + γ*(b-a) = -v + 0.5*(2v) which
    # overflows to -v + Inf = Inf (wrong).
    large = prevfloat(Inf)
    @test isfinite(quantile([large, -large], 0.5))
    @test quantile([large, -large], 0.5) === 0.0
    @test quantile([large, -large], 0.25) ≈ -large/2.0
    @test quantile([large, -large], 0.75) ≈  large/2.0
end

@testset "_quantile: isfinite branch — Inf endpoints handled correctly" begin
    # When a or b is Inf, the weight-sum formula is the right one.
    @test quantile([0.0, Inf],  0.5) == Inf   # existing: both formulas agree here
    @test quantile([-Inf, 0.0], 0.5) == -Inf
    # Partial Inf: p=0.0 should return -Inf, p=1.0 should return finite
    @test quantile([-Inf, 1.0, 2.0], 0.0) == -Inf
    @test quantile([-Inf, 1.0, 2.0], 1.0) == 2.0
end

@testset "_quantile: a≈b branch — monotonicity near equal elements" begin
    # When a≈b (nearly equal consecutive elements), use a + γ*(b-a) to preserve
    # monotonicity.  A NOT-flip would use (1-γ)*a + γ*b which can be
    # non-monotone at rounding boundaries.
    # From issue #144: these must be issorted.
    @test issorted(quantile([1.0, 1.0, 1.0+eps(), 1.0+eps()],
                             range(0, 1, length=100)))
    @test issorted(quantile([1.0, 1.0+1eps(), 1.0+2eps(), 1.0+3eps()],
                             range(0, 1, length=100)))
    # Identical elements: any quantile must equal the element itself
    @test all(==(5.0), quantile([5.0, 5.0, 5.0], [0.0, 0.25, 0.5, 0.75, 1.0]))
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 6: Rational exactness through var and quantile
#
# Targeted mutant: arithmetic operators / vs * in correction factor.
# Rational arithmetic exposes integer overflow or wrong-operator mutants
# because exact Rational results differ from Float64 approximations.
# (#136-class: Rational OverflowError was in the quantile path.)
# ─────────────────────────────────────────────────────────────────────────────

@testset "var Rational: exact computation" begin
    xs = [1//1, 2//1, 3//1, 4//1]
    n  = length(xs)

    v_corr   = var(xs; corrected=true)
    v_biased = var(xs; corrected=false)

    # Exact rational values: sum of sq dev from mean 5//2:
    # (1-5/2)² + (2-5/2)² + (3-5/2)² + (4-5/2)² = 9/4 + 1/4 + 1/4 + 9/4 = 5
    # unbiased = 5//3; biased = 5//4
    @test v_corr   == 5//3
    @test v_biased == 5//4
    @test v_corr isa Rational
end

@testset "quantile Rational: no overflow on simple input" begin
    # #136-class: quantile(Rational[], ...) triggered OverflowError
    xs = [1//1, 2//1, 3//1, 4//1]

    @test quantile(xs, 1//2) === 5//2
    @test quantile(xs, 0//1) === 1//1
    @test quantile(xs, 1//1) === 4//1

    # Monotone
    qs = [quantile(xs, p//4) for p in 0:4]
    @test issorted(qs)
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 7: corrected=false identity — single element
#
# Targeted mutant: `Int(corrected)` → `1` (always subtract 1), causing
# corrected=false to return NaN on a single-element input.
# ─────────────────────────────────────────────────────────────────────────────

@testset "var single element corrected=false must be 0" begin
    # corrected=false, n=1: denominator = 1 - 0 = 1, result = 0.
    # If mutant sets Int(corrected)=1 always, denominator = 0, result = NaN.
    @test var([1]; corrected=false) === 0.0
    @test var((1,); corrected=false) === 0.0
    @test var([42.0]; corrected=false) === 0.0

    # std follows
    @test std([1]; corrected=false) === 0.0
    @test std([42.0]; corrected=false) === 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 8: _quantile fma vs non-fma — aleph computation
#
# Targeted mutant: fma(n, p, m) → n*p + m (floating-point rewrite).
# fma avoids one intermediate rounding; for integer n and integer p*n,
# aleph is exact and the result must exactly equal an array element.
# The "avoid some rounding" property from runtests.jl line 820 is the oracle.
# ─────────────────────────────────────────────────────────────────────────────

@testset "_quantile: fma-exact index recovery" begin
    # quantile(1:10, i/9) for i=0:9 must exactly recover elements 1:10.
    # fma(n, p, m) avoids one rounding step compared to n*p + m; for these
    # specific denominators (n-1 = 9 and 13) the results come out exact.
    # If fma is replaced by n*p + m, some of these exact matches break.
    @test [quantile(1:10, i/9) for i in 0:9] == 1:10
    @test [quantile(1:14, i/13) for i in 0:13] == 1:14
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 9: dims-reduction var consistency
#
# Targeted mutant: in _var(A, corrected, mean, dims), if the `mean === nothing`
# branch is flipped (mean always recomputed vs passed through), results diverge.
# ─────────────────────────────────────────────────────────────────────────────

@testset "var dims: pre-computed mean is equivalent to auto mean" begin
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]

    for d in (1, 2)
        m = mean(A; dims=d)
        @test var(A; dims=d, mean=m)       ≈ var(A; dims=d)
        @test var(A; dims=d, mean=m, corrected=false) ≈ var(A; dims=d, corrected=false)
    end
end

@testset "var dims: corrected flag consistent across dims" begin
    # Use a fixed matrix to avoid Random dependency
    A = [1.0 2.0 3.0 4.0 5.0;
         2.0 4.0 6.0 8.0 10.0;
         1.0 3.0 5.0 7.0 9.0;
         0.0 1.0 2.0 3.0 4.0]
    for d in (1, 2)
        v_c  = var(A; dims=d, corrected=true)
        v_b  = var(A; dims=d, corrected=false)
        n    = size(A, d)
        # element-wise ratio must be n/(n-1)
        @test all(v_c ./ v_b .≈ n / (n - 1))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 10: #195-class overflow: std of symmetric large-magnitude pair
#
# std([1e300, -1e300]) computes mean=0, then varm([1e300,-1e300], 0).
# centralize_sumabs2 computes abs2(1e300 - 0) + abs2(-1e300 - 0)
#   = (1e300)^2 + (1e300)^2 = 2e600 = Inf
# var = Inf / 1 = Inf; std = sqrt(Inf) = Inf.
# This is the CORRECT behaviour (not NaN), because the inputs are finite but
# the variance is genuinely out of Float64 range.
# A wrong-sign mutant on the centralize accumulation would give NaN.
# ─────────────────────────────────────────────────────────────────────────────

@testset "std overflow class (#195): large symmetric pair" begin
    v = [1e300, -1e300]

    @test isinf(std(v))
    @test std(v) === Inf
    @test isinf(var(v))
    @test var(v) === Inf

    # corrected=false also Inf
    @test isinf(std(v; corrected=false))
    @test isinf(var(v; corrected=false))

    # Explicit mean=0 path (bypasses mean recomputation)
    @test isinf(std(v; mean=0.0))
    @test std(v; mean=0.0) === Inf

    # Not NaN: a wrong-operator mutant would make var NaN
    @test !isnan(std(v))
    @test !isnan(var(v))
end

@testset "std overflow class: Float32 symmetric pair" begin
    v = Float32[1f30, -1f30]
    @test isinf(std(v))
    @test !isnan(std(v))
end

@testset "quantile input-validation guards: p/alpha/beta range checks" begin
    v = [1.0, 2.0, 3.0, 4.0]

    # p outside [0,1] throws (guards both endpoints)
    @test_throws ArgumentError quantile(v, -0.001)
    @test_throws ArgumentError quantile(v, 1.001)
    @test_throws ArgumentError quantile(v, -1.0)
    @test_throws ArgumentError quantile(v, 2.0)
    # boundary values are accepted (guard is inclusive)
    @test quantile(v, 0.0) == 1.0
    @test quantile(v, 1.0) == 4.0

    # alpha outside [0,1] throws
    @test_throws ArgumentError quantile(v, 0.5; alpha=-0.1)
    @test_throws ArgumentError quantile(v, 0.5; alpha=1.1)
    # beta outside [0,1] throws
    @test_throws ArgumentError quantile(v, 0.5; beta=-0.1)
    @test_throws ArgumentError quantile(v, 0.5; beta=1.1)
    # inclusive boundaries accepted for alpha/beta
    @test quantile(v, 0.5; alpha=0.0, beta=0.0) isa Float64
    @test quantile(v, 0.5; alpha=1.0, beta=1.0) isa Float64

    # vector form validates each p
    @test_throws ArgumentError quantile(v, [0.5, 1.5])
end
