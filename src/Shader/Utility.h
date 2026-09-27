constant constexpr float PI = 3.14159265358979323846f;
constant constexpr float InvPI = 1.0f / PI;

constexpr float ConstexprSqrt(float x) {
    float curr = x >= 1.0f ? x : 1.0f, prev = 0.0f;
    while (curr != prev) { prev = curr; curr = 0.5f * (curr + x / curr); }
    return curr;
}

// ==============================================================================
// Atmospheric scattering coefficients
// ==============================================================================

constant constexpr float R_ground = 6360; // km
constant constexpr float R_top    = 6460;
constant constexpr float H        = ConstexprSqrt(R_top * R_top - R_ground * R_ground); // Maximum Geometric Horizon Distance