float wboitWeight(float z, float alpha) {
    float a = clamp(alpha, 0.0, 1.0);
    float z_scaled = abs(z) / 200.0;
    float z4 = z_scaled * z_scaled * z_scaled * z_scaled;
    return clamp(0.03 / (1e-5 + z4), 0.01, 3000.0) * a;
}

void wboitOutput(vec3 rgb, float alpha, float z, out vec4 accum, out vec4 reveal) {
    float a = clamp(alpha, 0.0, 1.0);
    float w = wboitWeight(z, a);
    accum = vec4(rgb * a * w, a * w);
    reveal = vec4(a);
}
