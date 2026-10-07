vec2 uvApply(vec4 m, vec4 o, vec2 uv) {
    // o.z is the existing unlit flag; o.w selects the texture-coordinate
    // set before rotation/scale/offset. The default UV0 path is unchanged.
    if (o.w > 0.5) uv = v_uv1;
    return vec2(m.x * uv.x + m.y * uv.y + o.x, m.z * uv.x + m.w * uv.y + o.y);
}
