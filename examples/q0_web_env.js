// Emscripten pre-run: browser query knobs reuse the native Q0 code path.
// Only the explicit graphics controls below are imported into ENV.
Module['preRun'] = Module['preRun'] || [];
Module['preRun'].push(function () {
    const query = new URLSearchParams(location.search);
    const knobs = {
        'frames': 'AGATE_Q0_FRAMES',
        'still': 'AGATE_Q0_STILL',
        'render-scale': 'AGATE_Q0_RENDER_SCALE',
        'ssr-pass': 'AGATE_Q0_SSR_PASS',
        'ssgi-pass': 'AGATE_Q0_SSGI_PASS',
        'ssgi-temporal': 'AGATE_Q0_SSGI_TEMPORAL',
        'no-ssr': 'AGATE_Q0_NO_SSR',
        'ao-mode': 'AGATE_Q0_AO_MODE',
        'ao-temporal': 'AGATE_Q0_AO_TEMPORAL',
        'no-ao': 'AGATE_Q0_NO_AO',
        'no-taa': 'AGATE_Q0_NO_TAA',
        'no-reactive': 'AGATE_Q0_NO_REACTIVE',
        'taa-blend': 'AGATE_Q0_TAA_BLEND',
        'taa-jitter': 'AGATE_Q0_TAA_JITTER',
        'no-dof': 'AGATE_Q0_NO_DOF',
        'no-shafts': 'AGATE_Q0_NO_SHAFTS',
        'no-ssgi': 'AGATE_Q0_NO_SSGI',
        'no-particles': 'AGATE_Q0_NO_PARTICLES',
        'no-glass': 'AGATE_Q0_NO_GLASS',
        'no-shadows': 'AGATE_Q0_NO_SHADOWS',
        'no-point-pcss': 'AGATE_Q0_NO_POINT_PCSS',
        'no-spot-pcss': 'AGATE_Q0_NO_SPOT_PCSS',
        'no-hemi': 'AGATE_Q0_NO_HEMI',
        'no-sun': 'AGATE_Q0_NO_SUN',
        'no-point': 'AGATE_Q0_NO_POINT',
        'no-spot': 'AGATE_Q0_NO_SPOT',
        'no-area': 'AGATE_Q0_NO_AREA',
        'no-bloom': 'AGATE_Q0_NO_BLOOM',
        'no-local-tonemap': 'AGATE_Q0_NO_LOCAL_TONEMAP',
        'camera-alpha': 'AGATE_Q0_CAMERA_ALPHA',
    };
    for (const [key, name] of Object.entries(knobs)) {
        if (query.has(key)) ENV[name] = query.get(key);
    }
});
