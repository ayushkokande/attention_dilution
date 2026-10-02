"""One inert filler passage; added requests and dialogue are separate experiments."""

BENIGN_SEED_PASSAGE = "The Apennine Mountains form the geological backbone of peninsular Italy, running\nsome 1,200 kilometres from Liguria in the north to Calabria in the south. They\nwere formed during the Cenozoic era through the convergence of the African and\nEurasian plates and continue to experience seismic activity today. The range is\ndivided into three principal sections: the Northern Apennines, the Central\nApennines, and the Southern Apennines, each with distinct geological histories.\nThe highest peak is Corno Grande in the Gran Sasso massif, rising 2,912 metres\nabove sea level. The mountains support a variety of ecosystems, from beech and\noak forests at lower elevations to alpine meadows above the treeline. Several\nnational parks protect this biodiversity, including Gran Sasso e Monti della\nLaga, Abruzzo Lazio e Molise, and Maiella. Wolves, brown bears, and chamois\ninhabit the more remote regions. Human settlement in the Apennines dates to\nprehistoric times, with hill towns perched on defensible ridges that have been\noccupied continuously for over a thousand years. Traditional pastoral economies\nof sheep transhumance shaped much of the cultural landscape, though most upland\nvillages have lost population to coastal cities since the mid-twentieth century.\nThe mountains also play a critical hydrological role, giving rise to many of\nItaly's major rivers including the Tiber, Arno, and Volturno. Snowfall feeds\nsprings that supply drinking water to several large urban populations. Climate\nvaries sharply with elevation and aspect, with Mediterranean conditions on the\nwestern slopes and a more continental regime on the Adriatic side. Winter\nsnowpack on the higher massifs persists into late spring, supporting limited\nski tourism in places such as Roccaraso and Campo Imperatore. The Apennines\nhave inspired writers, painters, and pilgrims for centuries, with monastic\nfoundations like Subiaco and Montecassino marking key sites in the religious\nand cultural development of medieval Europe.\n"


def build_filler(tokenizer, target_length, *, passage=BENIGN_SEED_PASSAGE):
    if target_length < 0:
        raise ValueError("Filler length cannot be negative")
    if target_length == 0:
        return ""
    seed = tokenizer(passage + "\n", add_special_tokens=False)["input_ids"]
    if not seed:
        raise ValueError("Filler passage produced no tokens")
    tokens = (seed * (target_length // len(seed) + 1))[:target_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def wrap_prompt(filler, request):
    return filler + "\n\n" + request if filler else request


def request_token_position(tokenizer, formatted_prompt, request, *, span=None):
    """Last token overlapping the request, before the assistant template suffix."""
    start = formatted_prompt.rfind(request) if span is None else span[0]
    if start < 0 or not request:
        raise ValueError("Request not found in formatted prompt")
    end = start + len(request) if span is None else span[1]
    if formatted_prompt[start:end] != request:
        raise ValueError("Target span does not match request")
    offsets = tokenizer(
        formatted_prompt, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    positions = [i for i, (lo, hi) in enumerate(offsets) if hi > start and lo < end]
    if not positions:
        raise ValueError("Request did not map to any tokens")
    return positions[-1]


def adaptive_batch_size(length):
    if length <= 1024:
        return 8
    if length <= 4096:
        return 4
    if length <= 16384:
        return 2
    return 1
