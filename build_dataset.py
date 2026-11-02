"""Build the GreenCart sample_submissions dataset + manifest (SPEC §6)."""
import json
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from gtts import gTTS
from imageio_ffmpeg import get_ffmpeg_exe
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

OUT = Path("sample_submissions")
CORPUS = Path("reference_corpus")
FFMPEG = get_ffmpeg_exe()

TEMPLATE = (
    "GreenCart Case Competition 2026: Problem Statement. GreenCart is a five-year-old online "
    "grocery delivery company operating in six metropolitan cities. It has 1.2 million monthly "
    "active customers and an average order value of 1,450 rupees, but its contribution margin per "
    "order is still negative in two of its six cities. The board is considering expanding into "
    "twelve tier-2 cities over the next eighteen months. Expansion would require an estimated "
    "investment of 380 crore rupees in dark stores, cold-chain logistics and marketing. "
    "Participants must answer three questions. First, should GreenCart expand now, delay, or not "
    "expand at all? Second, if it expands, which three cities should it enter first and why? "
    "Third, what operational changes are needed to make unit economics positive within two "
    "years? Support your answer with clear assumptions and numbers. Submissions are limited to "
    "four pages. Cite all external sources."
)

CROSSMODAL = (
    "Our recommendation is a phased entry, starting with Indore, Coimbatore and Jaipur. These "
    "three cities combine high smartphone penetration with dense residential clusters, which "
    "keeps delivery distances short. We estimate that a single dark store in each city can break "
    "even at around nine hundred orders per day, which is reachable within eight months if "
    "GreenCart reuses its existing supplier contracts."
)

QUICK_PARA = (
    "Quick commerce is a retail model in which online orders of groceries and household goods "
    "are delivered to customers within roughly thirty minutes. The format originated in India "
    "in the early 2020s and spread rapidly in dense urban neighbourhoods where orders can be "
    "fulfilled from small local warehouses positioned inside residential areas."
)

PITCH02 = (
    "We are GreenCart's pitch crew, and our answer is deliberately small. Enter two cities, not "
    "twelve, keep the burn under 90 crore rupees, and prove that one store model works before "
    "scaling it. Bhopal first, Kochi second, strictly in that order. If either store fails to "
    "break even within seven months, we stop expanding and ship the learnings back to the "
    "metro business."
)

# ---------------------------------------------------------------- bodies ---

ALPHA = [
    "We spent two weekends reading the pilot data before writing this, honestly. Jaipur ran 43 "
    "bikes in December and finished with a contribution of minus 11 rupees per order, which "
    "stung. But the same pilot crossed 900 orders a day in its final week, right after we added "
    "the two Diwali-season snack SKUs. Density fixed it, not discounting.",
    "Tier-2 rent is roughly a third of metro rent per square foot. Our Indore scouting trip "
    "found four dense residential clusters around Vijay Nagar where a two-wheeler can cover a "
    "delivery radius under 2.2 km. Cold storage is the gap; only one of the four sites has "
    "reliable backup power, so plan 60 lakh per city as a one-time fitout overhead.",
    "Recieve this or not, the competitive window is short. Two quick-commerce platforms have "
    "already entered Bhopal and Surat, and the first mover in a tier-2 city tends to hold "
    "somewhere near 55% category share for at least a year. Wait longer and we are buying "
    "share at double the CAC.",
    "Numbers we expect the panel to poke: phase one needs only 74 crore of the 380 crore "
    "envelope. Marketing at 180 rupees per acquired customer, retention holding near 68% at "
    "month three. If CAC payback stays under eleven months we go, full stop.",
]

BRAVO = [
    "Our case for expansion rests on three numbers. Online grocery penetration in tier-2 cities "
    "is 4.1% today against 11.8% in the metros, so the headroom is bigger than most people "
    "assume. Commercial land costs run at 40% of metro rates, and rider wages are 18% lower "
    "outside the six big cities.",
    "The catch is basket size. Our Kota pilot showed an average order value of 980 rupees "
    "against 1,450 in Delhi, because customers use the app as a top-up channel there. We "
    "compensate with a 25 rupee basket fee under 700 rupees, plus staples bundles, which "
    "lifted Kota's AOV by 21% in six weeks.",
    "Comitted plan: three cities in nine months, Bhopal, Vadodara and Dehradun. Each dark store "
    "needs roughly 1.1 crore of fitout plus 40 lakh of cold-chain capex, and local hiring of 34 "
    "riders and 11 store staff. Every site sits within 2.5 km of a wholesale mandi to shorten "
    "inbound replenishment.",
    "Per-store break-even lands near 520 orders a day by month eight in our model, sensitive "
    "mainly to fuel price and rider incentive. A 12% rise in diesel pushes that to month ten. "
    "We hold 15% contingency in the 380 crore ask for exactly this.",
    "We are ready, the sites are scouted and costed. One honest caveat: our cold-chain vendor "
    "has only operated in north India so far, so Vadodara should carry a separate vendor audit "
    "before the money is released for phase one.",
]

DELTA = [
    "We believe GreenCart should expand now, not later, and we will put real numbers behind "
    "that belief. Contribution margin across our six metro cities sits at minus 7 rupees per "
    "order. Fixing that through volume alone would need a 31% cut in rider cost, which is not "
    "realistic under any wage curve we modelled.",
    "Tier-2 expansion changes the cost line rather than the revenue line. A dark store in "
    "Bhubaneswar costs us 62 lakh to fit out against 1 crore 40 in Mumbai. Rent works out to "
    "14 rupees per order versus 39 in the metros, which is the single biggest lever we found.",
    "Our model says a tier-2 store needs 610 orders a day to break even. Our Pune flagship "
    "took nine months to reach 800. We expect roughly eight months in tier-2, because the "
    "catchment density around the three preferred sites exceeds 11,000 households per square "
    "kilometre.",
    "We picked Bhubaneswar, Nagpur and Lucknow. Each has at least one cluster of high-rise "
    "residential towers close to a wholesale mandi, and each city government has already "
    "approved commercial licences for food retail in those wards, which saves about seven "
    "weeks of paperwork.",
    "The operational fixes travel with us: renegotiate cold-chain contracts from a per-trip to "
    "a per-tonne basis, forecast demand in 15-minute buckets, and move packing back into the "
    "store from the central kitchen. We estimate 9 rupees of savings per order, enough to "
    "close the current gap entirely.",
]

ECHO = [
    "The case for moving now is the one we intend to defend, and we will argue it with figures. "
    "Per order, contribution margin stands at minus 7 rupees across the six metro locations. "
    "Curing that deficit through added volume alone would demand a 31% reduction in rider "
    "expense, and no wage curve we modelled makes that achievable.",
    "What expansion into tier-2 cities alters is the cost line, not the revenue line. Fitting "
    "out a dark store in Bhubaneswar takes 62 lakh, compared with 1 crore 40 in Mumbai. Rent "
    "comes to 14 rupees per order there, against 39 in the metros, and that spread is the "
    "largest lever our analysis surfaced.",
    "Break-even for a tier-2 store, per our model, arrives at 610 orders daily. Nine months "
    "were needed before our Pune flagship touched 800. Around eight months is our tier-2 "
    "expectation, since density in the catchments of the three preferred sites tops 11,000 "
    "households per square kilometre.",
    "Bhubaneswar, Nagpur and Lucknow are the three we settled on. Every one has a cluster of "
    "high-rise housing near a wholesale mandi, and each municipal government already grants "
    "food-retail trade licences in those wards, trimming some seven weeks off the paperwork.",
    "Our operational fixes come along with the rollout: shift cold-chain contracts from "
    "per-trip to per-tonne pricing, build forecasts on 15-minute demand buckets, and relocate "
    "packing from the central kitchen into the store. Of savings per order we expect about 9 "
    "rupees, sufficient to close the existing gap completely.",
]

FOXTROT = [
    "The proposed expansion into tier-2 cities represents a significant strategic opportunity "
    "for the company. It is important to note that the tier-2 retail market is projected to "
    "double in size within the next four years. A structured and phased entry would allow the "
    "company to leverage its existing supply contracts. This approach also minimises capital "
    "risk while establishing long-term brand presence.",
    "The recommended strategy prioritises a holistic evaluation of demand, logistics and "
    "competitive intensity across candidate markets. Cities such as Indore, Bhopal and Kochi "
    "combine robust consumption growth with moderate infrastructure costs. Furthermore, the "
    "availability of trained local staff reduces specialised onboarding requirements. A "
    "careful site-selection process would therefore improve the likelihood of early "
    "profitability.",
    "Financial modelling suggests that each dark store would require an investment of "
    "approximately one crore rupees. The expected contribution margin turns positive at a "
    "daily volume of five hundred orders. In this analysis, the median store reaches that "
    "threshold within ten months. Additionally, marketing expenses per acquired customer are "
    "projected at two hundred rupees.",
    "It is important to note that supply chain reliability determines the customer experience "
    "in online grocery retail. The company should establish regional consolidation centres to "
    "support the proposed distribution network. Furthermore, partnerships with established "
    "cold-chain vendors would ensure consistent product quality. These measures would create "
    "a robust operational foundation for subsequent growth phases.",
    "In conclusion, the company should proceed with a disciplined and phased expansion into "
    "tier-2 markets. A holistic framework that balances growth, cost and risk is recommended. "
    "The proposed approach leverages existing capabilities while building new regional "
    "strengths. This strategy offers a sustainable and durable path to long-term value "
    "creation.",
]

GOLF = [
    "We read the case, ran the numbers twice, and honestly we are voting to wait. Not forever, "
    "and not because we lack ambition. The arithmetic simply does not work yet.",
    "Right now our contribution margin is minus 31 rupees per order once cold-chain and "
    "packaging costs are counted (the case states minus 7, but that line excludes packaging; "
    "we added it back). At 1.2 million orders a month this is a 3.7 crore hole every month. "
    "Expansion would place three more such holes on the map.",
    "A tier-2 dark store still carries cold-chain capex of 45 lakh. Our vendor quoted rates 19% "
    "higher outside metro routes because container runs are thinner. That single line pushes "
    "store break-even to roughly 1,100 orders a day, higher than any tier-2 benchmark we "
    "located.",
    "Fix the core first: renegotiate the cold-chain contract, lift basket size by bundling "
    "staples, and take minus 31 to minus 5. Six months of honest cost work, then a re-look. "
    "If margins still have not improved by the second review, the decision has been made for "
    "us anyway.",
    "Two quarters. That is the whole pitch.",
]

HOTEL = [
    "We start with context on the sector, because the operating model of quick commerce drives "
    "every number that follows. " + QUICK_PARA,
    "That context matters for one reason: neighbourhood-level density is what makes "
    "thirty-minute delivery affordable. Our metro riders average 9.4 minutes per drop, while a "
    "test route in Kanpur covered 6.1 km per drop on broken roads, which burns an extra 22 "
    "rupees of fuel and 6 minutes of rider time per order.",
    "Our recommendation is selective expansion: Rajkot and Patna only, each fitout signalled at "
    "55 lakh, with a strict 700 rupee minimum basket in place until household density catches "
    "up. Both cities already have commercial cold-chain capacity routed through their fruit "
    "mandis, so inbound supply does not need a new third-party vendor.",
    "One more number for the sceptics on the panel. Our Patna scouting estimated 3,100 monthly "
    "orders per store at a modest 5% household penetration, against 2,400 needed per day to "
    "break even. It works, barely, and only under the strict two-city cap we are proposing.",
]

INDIA = [
    "Our answer is no, not this year. Expansion sounds exciting on a slide, but 380 crore is "
    "essentially our entire cash cushion, and we would be betting all of it on consumers we "
    "have never met.",
    "Look at the churn inside what we already run. 71% of customers place two or fewer orders "
    "after their first purchase, and only 9% stay past month four. Spending crores to find "
    "more such customers before fixing retention is backwards.",
    "The tier-2 consumer profile differs as well. A survey of 400 households in Jabalpur found "
    "62% shopping daily from local kiranas at roughly 80 rupees per trip. Our conjoint cards "
    "put the value of a thirty-minute delivery promise below 15 rupees for them.",
    "Meanwhile the six existing cities still need work. Fix the 26% out-of-stock rate, "
    "renegotiate packaging, which costs 14 rupees and mostly ends up in the bin, and lift "
    "average order value from 1,450 toward 1,600. Then talk about twelve cities.",
    "Cash buys options. The market will still be there in eighteen months; a balance sheet "
    "survives only so many ill-funded expansions.",
]

PITCH_REFINE = [
    "Final note before the board vote: the pilot taught us that operations beat advertising in "
    "the first year. We spent 1.9 crore on brand in Jaipur and got 0.4% of it back in orders, "
    "while a 22 rupee refill incentive on staples moved weekly repeat purchase by 9%. Spend "
    "phase one on picking labour and cold storage, not on hoardings.",
]

MANIFEST = {
    "files": {
        "sub01_alpha.pdf":   {"label": "original_with_template", "expect": {"status": "ok", "peer": True,  "external": False, "ai": False}},
        "sub02_bravo.pdf":   {"label": "original_with_template", "expect": {"status": "ok", "peer": True,  "external": False, "ai": False}},
        "sub03_charlie.docx": {"label": "exact_copy",            "expect": {"status": "ok", "peer": True,  "external": False, "ai": False}},
        "sub04_delta.docx":  {"label": "original_with_template", "expect": {"status": "ok", "peer": True,  "external": False, "ai": False}},
        "sub05_echo.docx":   {"label": "paraphrase",             "expect": {"status": "ok", "peer": True,  "external": False, "ai": False}},
        "sub06_foxtrot.pdf": {"label": "ai_generated",           "expect": {"status": "ok", "peer": False, "external": False, "ai": True}},
        "sub07_golf.pdf":    {"label": "original",               "expect": {"status": "ok", "peer": False, "external": False, "ai": False}},
        "sub08_hotel.docx":  {"label": "external_copy",          "expect": {"status": "ok", "peer": False, "external": True,  "ai": False}},
        "sub09_india.docx":  {"label": "original",               "expect": {"status": "ok", "peer": False, "external": False, "ai": False}},
        "sub10_juliet.pdf":  {"label": "degraded_empty",         "expect": {"status": "skipped"}},
        "pitch01_kilo.mp4":  {"label": "cross_modal_copy",       "expect": {"status": "ok", "peer": True,  "external": False, "ai": False}},
        "pitch02_lima.mp4":  {"label": "original",               "expect": {"status": "ok", "peer": False, "external": False, "ai": False}},
        "pitch03_mike.mp4":  {"label": "degraded_silent",        "expect": {"status": "skipped"}},
    },
    "pairs": [
        {"a": "sub02_bravo.pdf",  "b": "sub03_charlie.docx", "relation": "exact_copy"},
        {"a": "sub04_delta.docx", "b": "sub05_echo.docx",    "relation": "paraphrase"},
        {"a": "sub01_alpha.pdf",  "b": "pitch01_kilo.mp4",   "relation": "cross_modal"},
        {"a": "sub08_hotel.docx", "b": "quick_commerce.txt", "relation": "external"},
    ],
    "template_files": ["sub01_alpha.pdf", "sub02_bravo.pdf", "sub03_charlie.docx", "sub04_delta.docx", "sub05_echo.docx"],
    "note": "Overlap is flagged on BOTH sides of a copy pair; deciding who copied is a human decision.",
}


# --------------------------------------------------------------- writers ---

def write_pdf(name, flowables):
    style = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14)

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 810, "GreenCart Case Competition 2026 | Confidential")
        canvas.drawCentredString(A4[0] / 2, 28, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT / name), pagesize=A4)
    doc.build(flowables, onFirstPage=on_page, onLaterPages=on_page)


def make_docx(name, paras):
    doc = Document()
    for p in paras:
        doc.add_paragraph(p)
    doc.save(str(OUT / name))


def make_speech_mp4(name, script):
    with tempfile.TemporaryDirectory() as td:
        mp3 = str(Path(td) / "tts.mp3")
        gTTS(text=script, lang="en", tld="co.in").save(mp3)
        out = str(OUT / name)
        subprocess.run(
            [FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=10",
             "-i", mp3, "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", out],
            check=True, capture_output=True)


def make_silent_mp4(name):
    out = str(OUT / name)
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=10", "-t", "10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out],
        check=True, capture_output=True)


def main():
    OUT.mkdir(exist_ok=True)

    S = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14)

    def P(t):
        return Paragraph(t, S)

    # 1 sub01_alpha.pdf: template + body on p1, CROSSMODAL alone on page 2
    write_pdf("sub01_alpha.pdf",
              [P(TEMPLATE)] + [P(t) for t in ALPHA]
              + [PageBreak(), P(CROSSMODAL), P(PITCH_REFINE[0])])

    # 2 sub02_bravo.pdf
    write_pdf("sub02_bravo.pdf", [P(TEMPLATE)] + [P(t) for t in BRAVO])

    # 3 sub03_charlie.docx: template + identical body to sub02
    make_docx("sub03_charlie.docx", [TEMPLATE] + BRAVO)

    # 4 sub04_delta.docx / 5 sub05_echo.docx
    make_docx("sub04_delta.docx", [TEMPLATE] + DELTA)
    make_docx("sub05_echo.docx", [TEMPLATE] + ECHO)

    # 6 sub06_foxtrot.pdf: pure AI style, no template
    write_pdf("sub06_foxtrot.pdf", [P(t) for t in FOXTROT])

    # 7 sub07_golf.pdf
    write_pdf("sub07_golf.pdf", [P(t) for t in GOLF])

    # 8 sub08_hotel.docx
    make_docx("sub08_hotel.docx", [TEMPLATE] + HOTEL)

    # 9 sub09_india.docx
    make_docx("sub09_india.docx", INDIA)

    # 10 sub10_juliet.pdf: zero bytes
    (OUT / "sub10_juliet.pdf").write_bytes(b"")

    # 11-12 speech videos
    make_speech_mp4("pitch01_kilo.mp4", CROSSMODAL)
    make_speech_mp4("pitch02_lima.mp4", PITCH02)

    # 13 silent video
    make_silent_mp4("pitch03_mike.mp4")

    # manifest
    (OUT / "dataset_manifest.json").write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    print("built", len(list(OUT.iterdir())), "files in", OUT)


if __name__ == "__main__":
    main()
