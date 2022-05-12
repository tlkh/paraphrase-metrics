import streamlit as st
import spacy
from paraphrase_metrics import metrics as pm
import time
import difflib

st.set_page_config(page_title="TextDiff Visualizer")

def render_single_para(paragraph, segment_info, prefix="a", other="b", gap=" "):
    # span (diff text) change from red to cyan
    span_diff_1 = """<span style="background-color:LightCoral;color:black;border-radius:2px;" onmouseover="chbg_"""
    span_diff_2 = """('cyan')" onmouseout="chbg_"""
    span_diff_3 = """('LightCoral')" id='"""
    # span (same text) change from green to cyan
    span_same_1 = """<span style="background-color:LightGreen;color:black;border-radius:2px;" onmouseover="chbg_"""
    span_same_2 = """('cyan')" onmouseout="chbg_"""
    span_same_3 = """('LightGreen')" id='"""
    segments = ["<p>",]
    for i, m in enumerate(segment_info):
        span1_id = prefix+"_"+str(i)+"_1"
        span1_id_other = other+"_"+str(i)+"_1"
        if i > 0:
            m_prev = segment_info[i-1]
            segment1 = span_diff_1 + span1_id + span_diff_2 + span1_id + span_diff_3 + span1_id + "'>" + paragraph[m_prev[0]+m_prev[1]:m[0]] + "</span>"
        else:
            segment1 = span_diff_1 + span1_id + span_diff_2 + span1_id + span_diff_3 + span1_id + "'>" + paragraph[:m[0]] + "</span>"
        span2_id = prefix+"_"+str(i)+"_2"
        span2_id_other = other+"_"+str(i)+"_2"
        segment2 = span_same_1 + span2_id + span_same_2 + span2_id + span_same_3 + span2_id + "'>" + paragraph[m[0]:m[0]+m[1]] + "</span>"
        highlighting_code = """<script>
        function chbg_"""+span1_id+"""(colour){
        document.getElementById('"""+span1_id+"""').style.backgroundColor=colour;
        document.getElementById('"""+span1_id_other+"""').style.backgroundColor=colour;
        }
        function chbg_"""+span2_id+"""(colour){
        document.getElementById('"""+span2_id+"""').style.backgroundColor=colour;
        document.getElementById('"""+span2_id_other+"""').style.backgroundColor=colour;
        }
        </script>"""
        segments += [highlighting_code, segment1, segment2]
    segments.append("</p>")
    return gap.join(segments)

def render_diff(a_parapgraph, b_parapgraph, gap=" ", prefix=None):
    if prefix is None:
        prefix = str(int(time.time()))
    s = difflib.SequenceMatcher(None, a_parapgraph.lower(), b_parapgraph.lower(), autojunk=False)
    matching_blocks = s.get_matching_blocks()
    # a 
    a_segment_info = [[b.a,b.size] for b in matching_blocks]
    a_html_paragraph = render_single_para(a_parapgraph, a_segment_info, gap=gap, prefix=prefix+"_a", other=prefix+"_b")
    # b
    b_segment_info = [[b.b,b.size] for b in matching_blocks]
    b_html_paragraph = render_single_para(b_parapgraph, b_segment_info, gap=gap, prefix=prefix+"_b", other=prefix+"_a")
    # table
    table = """<table style="width:100%;font-family:sans-serif;font-size:large;"><tr style="background-color:white;padding=1px;">
        <td style="border: 1px solid silver;padding:0.4em;border-radius:4px;">"""+a_html_paragraph+"""</td>
        <td style="border: 1px solid silver;padding:0.4em;border-radius:4px;">"""+b_html_paragraph+"""</td>
    </tr></table>"""
    return table

@st.cache(allow_output_mutation=True)
def load_model():
    nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_model()

st.markdown("### TextDiff Visualizer")

mode = st.selectbox("Input", ["Custom", "Examples"])

if mode == "Custom":
    col1, col2 = st.columns(2)
    with col1:
        text_A = st.text_area("Text 1", value="The findings  are  being  published July 1st in the Annals of Internal Medicine.")
    with col2:
        text_B = st.text_area("Text 2", value="The findings are published in the July 1st issue of the Annals of Internal Medicine.")
else:
    examples = st.radio("Examples", [
        "The top rate will go to 4.45 percent for all residents with taxable incomes above $500,000. ; For residents with incomes above $500,000, the income-tax rate will increase to 4.45 percent.",
        "However, prosecutors have declined to take criminal action against guards, though Fine said his inquiry is not finished. ; Prosecutors have declined to take criminal action against corrections officers, although Fine said his inquiry was not finished.",
        "In trading on the New York Stock Exchange, Kraft shares fell 25 cents to close at $32.30. ; Kraft's shares fell 25 cents to close at $32.30 yesterday on the New York Stock Exchange.",
        "An attempt last month in the Senate to keep the fund open for another year fell flat. ; An attempt to keep the fund open for another year fell flat in the Senate last month.",
        "Prisoners were tortured and executed -- their ears and scalps severed for souvenirs. ; They frequently tortured and shot prisoners, severing ears and scalps for souvenirs.",
        "American has laid off 6,500 of its flight attendants since Dec. 31. ; Since October 2001, American has laid off 6,149 flight attendants.",
        ])
    text_A, text_B = examples.split(" ; ")

st.markdown("Visualization")

html_viz = render_diff(text_A, text_B)

st.components.v1.html(html_viz)

dist = round(pm.edit_distance(text_A, text_B), 2)
bleu = round(pm.self_bleu(text_A, text_B), 2)
text_A, text_B = nlp(text_A), nlp(text_B)
wpd = round(pm.wpd(text_A, text_B), 2)
ld = round(pm.ld(text_A, text_B), 2)

metriccol1, metriccol2, metriccol3, metriccol4 = st.columns(4)
metriccol1.metric("WPD", wpd)
metriccol2.metric("LD", ld)
metriccol3.metric("Edit Dist.", dist)
metriccol4.metric("BLEU", bleu)

with st.expander("More info"):
    st.markdown("""**Explantion of Metrics**

* **WPD**: Word Position Deviation measures structural changes between two paraphrases
* **LD**: Lexical Deviation measures degree of vocabulary changes between two paraphrases
* **Edit Dist.**: Levenshtein edit distance 
* **BLEU**: SELF-BLEU score

For more information, see https://github.com/tlkh/paraphrase-metrics
    """)