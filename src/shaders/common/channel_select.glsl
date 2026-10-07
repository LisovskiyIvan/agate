float channelSelect(vec4 s, float lane) {
    if (lane < 0.5) return s.r;
    if (lane < 1.5) return s.g;
    if (lane < 2.5) return s.b;
    return s.a;
}
