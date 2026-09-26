# Аудио

> Путь: src/agate/audio.zig + src/agate/audio/ · Импорт: agate.audio (root.zig) · Потоки: главный поток (управление) + аудиопоток (renderFrames, 0 аллокаций).

## Что это

Модуль `audio` — весь звук движка: процедурный синтез коротких эффектов, декодированные клипы (WAV/MP3/OGG), потоковое воспроизведение длинных файлов (музыка, эмбиент), иерархия шин микширования, пространственное позиционирование (затухание, Doppler), DSP-эффекты (biquad-фильтры, Freeverb-подобная реверберация) и окклюзия через рейкаст физики.

Архитектурно это фасад `audio.zig`, реэкспортирующий листья `audio/` (свободные функции + тонкие форвардеры владельца `AudioEngine`; листья никогда не импортируют фасад — правило против циклов, задокументированное в шапке файла):

| Лист | Ответственность |
|---|---|
| `types.zig` | общий словарь: `BusId`, `AudioConfig`, `BusConfig`, `VoiceKind`/`Voice`, лимиты |
| `engine.zig` | владелец `AudioEngine`: поля, жизненный цикл, форвардеры во все листья |
| `commands.zig` | очередь команд голосов (lock-free SPSC ring, кража слотов) |
| `playback.zig` | `play`/`playClip`, one-shot хелперы, `PlayOptions`/`ClipPlayOptions`, `spatializeWith` |
| `buses.zig` | реестр шин и DAG маршрутизации (до 128 шин) |
| `effects.zig` | DSP фильтров шин, пул ревербераторов, окклюзия |
| `mixer.zig` | `renderFrames` + ядра синтеза, приватное состояние `ActiveVoice` |
| `streams.zig` | слоты стримов, `playSound*`/`playMusic*`, `MusicPlayOptions` |
| `clip.zig` | `AudioClip` + WAV-декодирование |
| `decode.zig` | MP3/Ogg декодеры (`DecodedAudio`, `DecodeError`) |
| `dsp.zig` | biquad + реверберация (`BiquadFilter`, `ReverbProcessor`) |
| `occlusion.zig` | рейкаст-окклюзия (`AudioOcclusionConfig`, `AudioOcclusionTracker`, `AudioEmitter`) |
| `stream.zig` | `AudioStream`: SPSC ring, декодеры Ogg/MP3/WAV, gapless/crossfade |

`AudioEngine` — обычная структура, принадлежащая вызывающему коду (обычно полю `Scene`). Аудиопоток вызывает только `renderFrames` с буфером `[]f32` (interleaved stereo); всё управление (создание шин, запуск голосов) идёт из главного потока через очередь команд и атомики — без мьютексов в realtime-пути.

## Быстрый старт

```zig
const agate = @import("agate");

// Движок живёт рядом со сценой; sample_rate — частота аудиоустройства.
var engine = agate.audio.AudioEngine.init(.{
    .master_volume = 0.8,
    .max_buses = 32,
});
engine.start();
defer engine.shutdown();

// Процедурный one-shot: удар с позицией в мире (моно→панорама+затухание).
engine.updateListener(camera_pos, camera_right);
engine.playImpact(hit_pos, impact_speed);

// Короткий декодированный клип (WAV целиком в памяти).
var clip = try agate.audio.AudioClip.fromWavFile(alloc, "assets/sfx/shot.wav");
defer clip.deinit(alloc);
engine.playClip(&clip, .{ .volume = 1.0 });

// Длинный файл — стриминг без загрузки целиком: gapless loop музыки.
engine.playMusic(.{
    // PlaySoundOptions: путь, громкость, loop, шина
    .loop = true,
    .volume = 0.7,
}, "assets/music/theme.ogg");

// Каждый кадр (главный поток): докачка стримов, сглаживание окклюзии.
engine.updateStreams(dt);

// Аудиоколбэк устройства (аудиопоток): только рендер, 0 аллокаций.
engine.renderFrames(device_buffer);
```

Пространственный звук через шину:

```zig
const sfx = engine.createSpatialBus("sfx", .{
    .volume = 1.0,
    .attenuation_model = .inverse,
    .min_distance = 2.0,
    .max_distance = 40.0,
    .rolloff = 1.0,
}).?;
engine.play(.{
    .kind = .thump,
    .bus = sfx,
    .position = enemy_pos,
    .volume = 0.9,
});
```

## API

### Типы и лимиты (`audio/types.zig`)

```zig
pub const BusId = enum(u8) { _, pub const invalid: BusId = @enumFromInt(0xFF); ... };
pub const AudioBus = BusId;
pub const default_max_buses: usize = 32;
pub const max_bus_capacity: usize = 128;
pub const max_buses: usize = default_max_buses;
pub const invalid_bus: BusId = BusId.invalid;

pub const AudioConfig = struct {
    max_buses: usize = default_max_buses,
    master_volume: f32 = 0.8,
    muted: bool = false,
};
pub const AudioEngineConfig = AudioConfig;

pub const AttenuationModel = enum(u8) { linear = 0, inverse = 1, exponential = 2 };
pub const BusAttenuation = struct {
    model: AttenuationModel = .inverse,
    min_distance: f32 = 1.0,
    max_distance: f32 = 30.0,
    rolloff: f32 = 1.0,
};
```

Лимиты движка (переэкспортированы и как вложенные константы `AudioEngine.max_voices` и т.д.):

| Константа | Значение | Смысл |
|---|---|---|
| `max_voices` | 24 | одновременно звучащих процедурных голосов |
| `max_distance` | 30.0 | дистанция полного затухания по умолчанию (м) |
| `max_commands` | 64 | глубина очереди команд голосам (SPSC ring) |
| `max_reverbs` | 4 | пул ревербераторов на движок |
| `chunk_frames` / `chunk_samples` | 64 / 128 | квант рендера микшера |
| `max_streams` | 32 | одновременно открытых стримов |

`VoiceKind` — `noise_burst | thump | blip | sample`. `Voice` — слот процедурного голоса: `active`, `kind`, `bus: ?BusId`, огибающая (`t`, `duration`, `volume`), свип частоты (`freq`, `freq_end`), свип фильтра (`cutoff`, `cutoff_end`), `pan`, `seed`/`lp`/`phase` для нойза, плюс состояние сэмпла (`clip: ?*const AudioClip` — только чтение, должен пережить голос; `sample_pos: f64`, `sample_step`, `loop`).

### Жизненный цикл (`audio/engine.zig`)

```zig
pub fn init(config: AudioConfig) AudioEngine
pub fn configure(self: *AudioEngine, config: AudioConfig) void
pub fn setBusCapacity(self: *AudioEngine, cap: usize) void
pub fn getBusCapacity(self: *const AudioEngine) usize
pub fn start(self: *AudioEngine) void
pub fn shutdown(self: *AudioEngine) void
pub fn stopAll(self: *AudioEngine) void
pub fn setMuted(self: *AudioEngine, muted_val: bool) void
pub fn toggleMuted(self: *AudioEngine) void
pub fn isMuted(self: *const AudioEngine) bool
pub fn setMasterVolume(self: *AudioEngine, vol: f32) void
pub fn getMasterVolume(self: *const AudioEngine) f32
pub fn updateListener(self: *AudioEngine, pos: Vec3, right: Vec3) void
pub fn updateListenerWithVelocity(self: *AudioEngine, pos: Vec3, right: Vec3, vel: Vec3) void
```

`init` не трогает устройство — только поля; `start` поднимает флаг работы, `shutdown` гасит голоса и стримы. Слушатель задаётся позицией и правым вектором камеры (forward выводится); вариант с `vel` нужен для Doppler-расчёта.

### Процедурное воспроизведение (`audio/playback.zig`)

```zig
pub const PlayOptions = struct { // поля: kind, bus, pan, position/min/max/rolloff/model/doppler/velocity, volume, duration, freq/freq_end, cutoff/cutoff_end, pitch, pitch_randomness, occlusion, occlusion_config
    kind: VoiceKind = .thump, bus: ?BusId = null, pan: f32 = 0.0,
    position: ?Vec3 = null, min_distance: ?f32 = null, max_distance: ?f32 = null,
    rolloff: ?f32 = null, attenuation_model: ?AttenuationModel = null,
    doppler_factor: ?f32 = null, velocity: ?Vec3 = null,
    volume: f32 = 0.5, duration: f32 = 0.2,
    freq: f32 = 110.0, freq_end: f32 = 55.0,
    cutoff: f32 = 5000.0, cutoff_end: f32 = 500.0,
    pitch: f32 = 1.0, pitch_randomness: f32 = 0.0,
    occlusion: f32 = 0.0, occlusion_config: AudioOcclusionConfig = .{},
};
pub const SpatializeResult = struct { ... };
pub fn play(self: anytype, params: PlayOptions) void
pub fn playImpact(self: anytype, position: Vec3, speed: f32) void
pub fn playImpactOn(self: anytype, bus: ?BusId, position: Vec3, speed: f32) void
pub fn playExplosion(self: anytype, position: Vec3, size: f32) void
pub fn playExplosionOn(self: anytype, bus: ?BusId, position: Vec3, size: f32) void
pub fn playBlip(self: anytype, freq: f32) void
pub fn playBlipOn(self: anytype, bus: ?BusId, freq: f32) void
pub const ClipPlayOptions = struct { // как PlayOptions, но: volume = 1.0, loop, rate, без duration/freq/cutoff/pitch
    bus: ?BusId = null, pan: f32 = 0.0, position: ?Vec3 = null, ...,
    volume: f32 = 1.0, loop: bool = false, rate: f32 = 1.0,
    pitch_randomness: f32 = 0.0, occlusion: f32 = 0.0, ...
};
pub fn playClip(self: anytype, clip: *const AudioClip, options: ClipPlayOptions) void
pub fn spatializeWith(...) SpatializeResult
```

`null` в полях дистанции/затухания означает «наследовать от шины». `play` не возвращает handle — это fire-and-forget: команда кладётся в очередь (`commands.pushCommand`), свободный голос ищется в аудиопотоке, при переполнении крадётся самый старый/тихий. `playImpact`/`playExplosion`/`playBlip` мапят игровые параметры (скорость удара, размер взрыва) на `PlayOptions`; варианты `*On` принимают явную шину.

### Шины (`audio/buses.zig`)

```zig
pub fn createBus(self: *AudioEngine, config: BusConfig) ?BusId
pub fn createSpatialBus(self: *AudioEngine, name: []const u8, config: ?BusConfig) ?BusId
pub fn createNonSpatialBus(self: *AudioEngine, name: []const u8, volume: f32) ?BusId
pub fn configureBus(self: *AudioEngine, bus: BusId, config: BusConfig) void
pub fn destroyBus(self: *AudioEngine, bus: BusId) void
pub fn findBus(self: *const AudioEngine, name: []const u8) ?BusId
pub fn findOrCreateBus(self: *AudioEngine, name: []const u8, config: BusConfig) ?BusId
pub fn isBusActive(self: *const AudioEngine, bus: BusId) bool
pub fn getBusCount(self: *const AudioEngine) usize
pub fn isBusSpatial(...) bool / pub fn setBusSpatial(self: *AudioEngine, bus: BusId, spatial: bool) void
pub fn getBusName / pub fn setBusName(self: *AudioEngine, bus: BusId, name: []const u8) void
pub fn setBusParent(self: *AudioEngine, bus: BusId, parent: ?BusId) void
pub fn getBusParent(self: *const AudioEngine, bus: BusId) ?BusId
pub fn setBusAttenuation(self: *AudioEngine, bus: BusId, model: AttenuationModel, min_dist: f32, max_dist: f32, rolloff: f32) void
pub fn getBusAttenuation(self: *const AudioEngine, bus: BusId) BusAttenuation
pub fn setBusDopplerFactor / pub fn getBusDopplerFactor(...) f32
pub fn setBusVolume / pub fn getBusVolume(...) f32
pub fn getBusEffectiveVolume(self: *const AudioEngine, bus_opt: ?BusId) f32
pub fn setBusMuted / pub fn isBusMuted / pub fn toggleBusMuted
pub fn stopBus(self: *AudioEngine, bus: BusId) void
```

Шина — именованный узел маршрутизации с флагами spatial/volume/mute, родителем (`setBusParent` строит DAG; эффективная громкость — произведение цепочки родителей, mute — каскадный) и настройками затухания/Doppler. `createBus` возвращает `null` при исчерпании ёмкости (`max_bus_capacity = 128` жёсткий предел, `AudioConfig.max_buses` — мягкий, меняется через `setBusCapacity`). `stopBus` гасит все голоса и стримы шины. Непространственные шины (`createNonSpatialBus`) игнорируют позицию — для музыки/UI.

### Эффекты шин (`audio/effects.zig`, `audio/dsp.zig`)

```zig
// dsp.zig
pub const BiquadFilterType = enum(u8) { ... }; // none + lowpass/highpass/bandpass/...
pub const BusFilterConfig = struct { ... };
pub const BusReverbConfig = struct { ... };
pub const BiquadFilter = struct {
    pub fn setParams(self: *BiquadFilter, filter_type: BiquadFilterType, cutoff_hz: f32, q_val: f32, sample_rate: f32) void
    pub fn processBuffer(self: *BiquadFilter, buffer: []f32) void
    pub fn resetState(self: *BiquadFilter) void
};
pub const ReverbProcessor = struct { // comb + allpass сеть, Freeverb-стиль
    pub fn init(sample_rate: f32) ReverbProcessor
    pub fn setSampleRate(self: *ReverbProcessor, rate: f32) void
    pub fn setConfig(self: *ReverbProcessor, config: BusReverbConfig) void
    pub fn processBuffer(self: *ReverbProcessor, buffer: []f32) void
    pub fn clear(self: *ReverbProcessor) void
};

// effects.zig (методы движка)
pub fn setBusFilter(self: *AudioEngine, bus: BusId, filter_type: BiquadFilterType, cutoff: f32, q: f32) void
pub fn setBusFilterCutoff / pub fn setBusFilterQ / pub fn clearBusFilter / pub fn getBusFilter
pub fn setBusReverb(self: *AudioEngine, bus: BusId, config: BusReverbConfig) bool // false = пул из 4 занят
pub fn clearBusReverb / pub fn getBusReverb / pub fn isBusReverbEnabled
pub fn setBusUnderwater / pub fn setBusMuffled / pub fn setBusTelephone // пресеты фильтра
pub fn setBusCaveReverb / pub fn setBusRoomReverb // пресеты реверба, bool как setBusReverb
```

Фильтр — один biquad (TDF-II, стерео-раздельные delay-регистры) на шину; cutoff клампится к Найквисту `[10, rate*0.499]`, Q — к `[0.1, 20]`. Пресеты: underwater (низкий lowpass), muffled, telephone (bandpass). Ревербераторы берутся из пула (`max_reverbs = 4`): `setBusReverb` возвращает `false`, если свободных нет.

### Окклюзия (`audio/occlusion.zig`)

```zig
pub const AudioOcclusionConfig = struct {
    min_volume: f32 = 0.25,   // громкость при 100% окклюзии (0 = тишина)
    min_cutoff: f32 = 500.0,  // lowpass при 100% окклюзии, Гц
    max_cutoff: f32 = 20000.0,// lowpass без окклюзии, Гц
    num_rays: u8 = 1,         // 1 = прямой луч; 3..5 = multi-tap с разбросом
    spread_radius: f32 = 0.6, // радиус разброса тапов, м
    smooth_time: f32 = 0.15,  // постоянная сглаживания, с
};
pub const RaycastFn = *const fn (origin: Vec3, direction: Vec3, max_distance: f32, user_data: ?*anyopaque) bool;
pub fn evaluateRaycastOcclusion(listener_pos: Vec3, emitter_pos: Vec3, config: AudioOcclusionConfig, raycast_fn: RaycastFn, user_data: ?*anyopaque) f32
pub const AudioOcclusionTracker = struct { ... }; // временное сглаживание 0..1
pub const AudioEmitter = struct { ... };

// Методы движка (effects.zig):
pub fn setBusOcclusion(self: *AudioEngine, bus: BusId, occlusion: f32) void
pub fn getBusOcclusion / pub fn setBusOcclusionConfig / pub fn getBusOcclusionConfig / pub fn clearBusOcclusion
pub fn updateBusOcclusion(self: *AudioEngine, bus: BusId, target_occlusion: f32, dt: f32) void
pub fn updateBusOcclusionWithRaycast(...) // evaluate + update за один вызов
```

Окклюзия возвращает фактор `0` (чисто) — `1` (закрыто); применяется как аттенюатор громкости (`min_volume`) плюс схлопывание lowpass (`min_cutoff`…`max_cutoff`). Multi-tap — центр + до 4 ортогональных тапов вокруг эмиттера (эмуляция дифракции на краях); базис строится от направления, с фолбэком при взгляде строго вверх. Рейкаст поставляет игра (обычно физика, см. `./physics.md`): функция возвращает `true`, если луч упёрся в препятствие. Сглаживание через `smooth_time` убирает щелчки при резком изменении видимости.

### Клипы и декод (`audio/clip.zig`, `audio/decode.zig`)

```zig
pub const AudioClip = struct {
    pub fn frameCount(self: *const AudioClip) usize
    pub fn deinit(self: *AudioClip, allocator: std.mem.Allocator) void
    pub fn fromMp3Memory(allocator: std.mem.Allocator, bytes: []const u8) decode.DecodeError!AudioClip
    pub fn fromMp3File(allocator: std.mem.Allocator, path: []const u8) !AudioClip
    pub fn fromOggMemory / pub fn fromOggFile(...) ...
    pub fn fromWavFile(allocator: std.mem.Allocator, path: []const u8) !AudioClip
    pub fn fromWavMemory(allocator: std.mem.Allocator, bytes: []const u8) !AudioClip
};
pub const DecodedAudio = struct { ... };
pub const DecodeError = error{ ... };
pub fn decodeMp3Memory / decodeMp3File / decodeOggMemory / decodeOggFile(...)
```

Клип — полностью декодированный PCM-блок в памяти (для коротких SFX). Декодирование аллоцирует через переданный аллокатор и происходит в главном потоке, не в аудиоколбэке. Длинные файлы так не грузят — для них стриминг ниже.

### Стриминг (`audio/stream.zig`, `audio/streams.zig`)

```zig
pub const StreamFormat = enum { auto, ogg, mp3, wav };
pub const StreamState = enum(u8) { ... }; // stopped/playing/paused/...
pub const StreamError = error{ ... };
pub const StreamOptions = struct {
    bus: ?BusId = null, volume: f32 = 1.0, pan: f32 = 0.0,
    loop: bool = false, buffer_frames: usize = AudioStream.default_buffer_frames,
    start_paused: bool = false, format: StreamFormat = .auto,
    fade_in_time: f32 = 0.0, auto_destroy: bool = false,
};
pub const PlaySoundOptions = StreamOptions;
pub const AudioStream = struct {
    pub const default_buffer_frames: usize = 65536; // ~1.48 c при 44.1 кГц
    pub fn openFile(allocator: std.mem.Allocator, path: []const u8, engine_sample_rate: f32, options: StreamOptions) StreamError!*AudioStream
    pub fn openMemory(allocator: std.mem.Allocator, bytes: []const u8, format: StreamFormat, engine_sample_rate: f32, options: StreamOptions) StreamError!*AudioStream
    pub fn deinit(self: *AudioStream) void
    pub fn play / pub fn pause / pub fn unpause / pub fn @"resume" / pub fn stop
    pub fn seekToSeconds(self: *AudioStream, seconds: f32) void
    pub fn seekToFrame(self: *AudioStream, frame: usize) void
    pub fn setVolume / pub fn getVolume / pub fn setPan / pub fn getPan
    pub fn setBus(self: *AudioStream, bus: ?BusId) void
    pub fn getBus(self: *const AudioStream) ?BusId
    pub fn setLoop / pub fn isLooping
    pub fn fadeTo(self: *AudioStream, target_volume: f32, duration_seconds: f32, stop_on_fade_out: bool) void
    pub fn getState(self: *const AudioStream) StreamState
};

// Уровень движка (streams.zig):
pub fn registerStream(self: *AudioEngine, stream: *AudioStream) !void
pub fn unregisterStream(self: *AudioEngine, stream: *AudioStream) void
pub fn createStreamFromFile / pub fn createStreamFromMemory(...)
pub fn destroyStream(self: *AudioEngine, stream: *AudioStream) void
pub fn updateStreams(self: *AudioEngine, dt: f32) void
pub fn playSound(...) / pub fn playSoundFromMemory(...)       // SFX-стрим
pub fn playSoundOnce / pub fn playSoundOnceFromMemory(...)    // one-shot с auto_destroy
pub fn stopSound(self: *AudioEngine, stream: *AudioStream, fade_duration: f32) void
pub fn pauseSound / pub fn resumeSound / pub fn setSoundVolume / pub fn stopAllSounds
pub fn crossfadeSound / pub fn crossfadeSoundFromMemory(...)  // кроссфейд двух стримов
pub fn playMusic / pub fn playMusicFromMemory / pub fn crossfadeMusic(...)
pub fn stopMusic(self: *AudioEngine, fade_duration: f32) void
pub fn pauseMusic / pub fn resumeMusic / pub fn setMusicVolume
pub fn getMusicStream(self: *const AudioEngine) ?*AudioStream
pub const MusicPlayOptions = struct { bus: ?BusId = null, volume: f32 = 1.0, loop: bool = true, rate: f32 = 1.0 };
pub fn playMusicClip(self: *AudioEngine, clip: *const AudioClip, options: MusicPlayOptions) void
```

Стрим: декодер (Ogg/MP3/WAV, формат резолвится по пути или задан явно) пишет в SPSC ring (`capacity` — степень двойки, branchless-индексы через маску), аудиопоток только читает. `refill` (докачка) вызывается из `updateStreams` в главном потоке. `fadeTo` с `stop_on_fade_out = true` даёт `stopSound(fade)` и `crossfade*` без кликов. `playSoundOnce*` ставит `auto_destroy` — стрим освобождается сам по EOF. `playMusicClip` останавливает шину и запускает клип как музыку (удобно для зацикленных тем, уже лежащих в памяти).

### Миксер (`audio/mixer.zig`)

```zig
pub fn renderFrames(self: anytype, buffer: []f32) void
```

Единственная realtime-точка. Принимает interleaved-stereo буфер, дренирует очередь команд, продвигает 24 голоса чанками по 64 фрейма, смешивает стримы из ring-буферов, применяет фильтры/реверб/мастер-громкость. Сложность O(voices + streams) на чанк; аллокаций ноль, блокировок ноль (только SPSC-индексы и атомики громкости/панорамы).

## Потоки и владение

- **Владение.** `AudioEngine` — value-struct у владельца (сцены). Клипы владеют PCM (`deinit` с тем же аллокатором). `AudioStream` создаётся аллокатором (`openFile`/`openMemory`/`createStreamFrom*`) и освобождается через `deinit`/`destroyStream`; `auto_destroy`-стримы освобождаются сами. `clip` в `Voice` и `ClipPlayOptions` — заимствованный указатель, должен пережить звучание.
- **Главный поток:** всё создание/конфигурирование (шины, пресеты, `play*`, `updateStreams`, `updateBusOcclusion*`). `updateStreams(dt)` обязателен каждый кадр, иначе ring опустеет и стрим захлебнётся.
- **Аудиопоток:** только `renderFrames`. Запрещены аллокации, файловый I/O, мьютексы, вызовы декодеров. Очередь команд (`max_commands = 64`) — единственный канал «главный → аудио» для голосов; переполнение молча отбрасывает новые команды (см. краевые случаи).
- `volume`/`pan`/`bus_id`/`loop` стрима — атомики (`std.atomic.Value`), безопасно менять из главного потока во время звучания.

## Ошибки и краевые случаи

- `createBus` → `null` при исчерпании ёмкости; `findOrCreateBus` не создаёт сверх лимита. Проверяйте `?BusId` до использования — несуществующая шина игнорируется тихо.
- Очередь команд 64: спам `play` в одном кадре сверх лимита теряет хвост. Для плотных очередей (пулемёт, дождь) дросселируйте на стороне игры.
- Голосов 24: при переполнении крадётся существующий (старый/тихий). Длинные громкие звуки могут быть вытеснены — разносите музыку и эмбиент в стримы, а не в голоса.
- `setBusReverb` / `setBusCaveReverb` / `setBusRoomReverb` → `false`, когда пул из 4 ревербераторов занят. Освобождайте через `clearBusReverb`.
- `updateBusOcclusionWithRaycast` с `dist < 1e-4` (эмиттер в точке слушателя) возвращает 0 без рейкаста. При `num_rays > 5` клампится к 5 тапам.
- `AudioStream.openMemory` требует явный `StreamFormat` (автоопределение по пути недоступно); `openFile` с неизвестным расширением и `.auto` → `error.InvalidFormat`.
- `seekToFrame` за концом клиппится к `total_frames`; `seekToSeconds` пересчитывается через `engine_rate`.
- Родительские циклы шин (`setBusParent` друг на друга) — ответственность вызывающего; движок циклы не детектирует, эффективная громкость может обнулиться/зациклиться в вычислении.

## Производительность

- Аудиоколбэк: 0 аллокаций, 0 локов; чанки 64 фрейма держат инвариант realtime. Biquad — TDF-II, ~5 MAC на сэмпл; реверб — фиксированная comb/allpass-сеть на шину (только на шинах с включённым ревербом, максимум 4).
- Hi-Z здесь нет — окклюзия стоит столько, сколько стоит пользовательский рейкаст × число тапов (1–5) × число эмиттеров; вызывайте `updateBusOcclusionWithRaycast` не чаще нескольких раз в секунду на эмиттер либо при движении, полагаясь на `smooth_time`.
- Стримы: ring по умолчанию 65536 фреймов (~1.5 с) — запас против джиттера `updateStreams`; уменьшайте `buffer_frames` (степень двойки) для экономии памяти на коротких SFX-стримах.
- Декодирование MP3/Ogg — только в главном потоке при открытии/предзаполнении; в колбэке декодеров нет, только memcpy из ring + ресемплинг.

## Смотрите также

- `./physics.md` — рейкаст для окклюзии (`RaycastFn` обычно implemented поверх физмира).
- `./scene.md` — владение `AudioEngine` сценой, `updateStreams` в кадре.
- `./profiler.md` — замер аудиофаз при просадках (update/physics/prepare раскладка).
- `./assets.md` — загрузка аудиофайлов как ассетов (пути для `playSound`/`playMusic`).
- `./serialization.md` — что из аудиосостояния (не) сохраняется в снапшот сцены.
