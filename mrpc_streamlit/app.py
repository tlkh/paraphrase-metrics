import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

with st.sidebar.expander("📍 Explanation", expanded=False):
    st.markdown("""
    **About**

    This demo allows you to explore the data inside the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset.
    It illustrates how **Word Position Deviation (WPD)** and **Lexical Deviation (LD)** can be used to find different types of [paraphrase pairs](https://direct.mit.edu/coli/article/39/3/463/1434/What-Is-a-Paraphrase) inside MRPC.
    By using what we observe from the data, we can find and correct numerous labelling errors inside MRPC, thus we present a revision of MRPC termed as **MRPC-R1**.

    **Data Display**

    The paraphrase pairs are displayed as **S1** and **S2** from the original MRPC (columns 1,2) and MRPC-R1 (columns 3,4), along with their labels (columns 5), showing if the label was changed or kept.

    By changing the **Display Types** option below, you can filter the displayed pairs to show pairs that were rejected (label changed from paraphrase to non-paraphrase) or corrected (inconsistencies corrected).

    This demo accompanies the paper ["Towards Better Characterization of Paraphrases" (ACL 2022)](https://openreview.net/forum?id=t2UJIFZVyz4), which describes in detail the methodologies used.""")

with st.sidebar.expander("⚙️ Dataset Options", expanded=False):
    st.markdown("This allows you to switch between the MRPC train and test sets, as well as choose to display only the original paraphrase pairs (MRPC) and/or the corrected pairs (MRPC-R1).")
    split = st.radio("Dataset Split", ["train", "test"])
    display = st.radio("Display only pairs from", [
        "Both MRPC and MRPC-R1", "Only MRPC", "Only MRPC-R1"])

ptype = st.sidebar.radio("Display Types", ["All Paraphrases",
                                           "Only Paraphrases in MRPC-R1",
                                           "Rejected Paraphrases from MRPC",
                                           "Corrected Paraphrases from MRPC"])

display_reason = st.sidebar.checkbox(
    "Display reason for label change", value=False)

with st.sidebar.expander("📍 Label Change Explanation", expanded=False):
    st.markdown("""
    Labels may change between MRPC and MRPC-R1, as displayed in column 5.

    For example, **1->0** means that the pair was labelled as a paraphrase (1) in MRPC, but corrected to non-paraphrase (0) in MRPC-R1, meaning we **rejected** the paraphrase.

    There are three main cases:

    1. **no need to correct**: label was accepted. The text in original pair and new pair is the **same**.
    2. **corrected**: label was kept as sentences were corrected. The text in original pair and new pair is **different**.
    3. **can't correct**: label was rejected as sentences could not be corrected. The text in original pair and new pair is the **same**.
    """)

st.sidebar.markdown("**WPD/LD Score Filter Options**")
display_range_wpd=st.sidebar.slider(
    "Filter by WPD Scores", min_value=0.0, max_value=1.0, value=(0.1, 0.7))
display_range_ld=st.sidebar.slider(
    "Filter by LD Scores", min_value=0.0, max_value=1.0, value=(0.1, 0.4))

with st.sidebar.expander("📍 WPD/LD Score Explanation", expanded=False):
    st.markdown("""
    WPD and LD measure differences in the two sentences of a paraphrase pair:

    * WPD measures difference in the sentence structure
    * LD measures differences in the words used

    By setting WPD to a high range (>0.4) and LD to a low range (<0.1), we can find paraphrases that do not change much in words used but have very different structures.

    When LD is set to a high range (>0.5), we can find many pairs labelled as paraphrases in MRPC are not in fact paraphrases.
    """)

    st.markdown("**Additional Filter Options**")

    filter_by=st.radio(
        "Filter By Scores From", ["MRPC", "MRPC-R1"])

    display_scores=st.checkbox("Display scores", value=False)


def load_df(split):
    if split == "train":
        df=pd.read_csv("./mrpc_train_scores.csv")
    else:
        df=pd.read_csv("./mrpc_test_scores.csv")
    df.reset_index(drop=True, inplace=True)
    return df


def filter_df(df, display, ptype, filter_by, display_scores):
    # filter data
    if display == "Only MRPC":
        df=df.drop(["new_s1", "new_s2"], axis=1)
    elif display == "Only MRPC-R1":
        df=df.drop(["og_s1", "og_s2"], axis=1)
    # filter paraphrase type
    if ptype == "All Paraphrases":
        condition=df.og_label == 1
        df_sel=df[condition]
    elif ptype == "Only Paraphrases in MRPC-R1":
        condition=df.new_label == 1
        df_sel=df[condition]
    elif ptype == "Rejected Paraphrases from MRPC":
        condition=(df.new_label == 0) & (df.og_label == 1)
        df_sel=df[condition]
    elif ptype == "Corrected Paraphrases from MRPC":
        condition=df.remarks == "corrected"
        df_sel=df[condition]
    else:
        # all
        df_sel=df
    # sort by scores
    if filter_by == "MRPC":
        # wpd
        condition=(df_sel.og_wpd >= display_range_wpd[0]) & (
            df_sel.og_wpd < display_range_wpd[1])
        df_sel=df_sel[condition]
        # ld
        condition=(df_sel.og_ld >= display_range_ld[0]) & (
            df_sel.og_ld < display_range_ld[1])
        df_sel=df_sel[condition]
    else:
        # wpd
        condition=(df_sel.new_wpd >= display_range_wpd[0]) & (
            df_sel.new_wpd < display_range_wpd[1])
        df_sel=df_sel[condition]
        # ld
        condition=(df_sel.new_ld >= display_range_ld[0]) & (
            df_sel.new_ld < display_range_ld[1])
        df_sel=df_sel[condition]
    # filter scores
    if filter_by == "MRPC":
        df_sel.sort_values("og_ld", inplace=True)
        df_sel.sort_values("og_wpd", inplace=True)
    else:
        df_sel.sort_values("new_ld", inplace=True)
        df_sel.sort_values("new_wpd", inplace=True)
    if not display_scores:
        df_sel.drop(["og_ld", "og_wpd", "new_ld", "new_wpd"],
                    axis=1, inplace=True)
    if not display_reason:
        df_sel.drop(["remarks", ],
                    axis=1, inplace=True)
    label_col=df_sel["og_label"].astype(
        str)+"->"+df_sel["new_label"].astype(str)
    df_sel["og/new label"]=label_col
    df_sel.drop(["og_label", "new_label"], axis=1, inplace=True)
    df_sel.drop_duplicates(inplace=True, ignore_index=True)
    return df_sel


df=load_df(split)

df_sel=filter_df(df, display, ptype, filter_by, display_scores)
df_sel.rename(columns={"og_s1": "Original S1 (MRPC)", "og_s2": "Original S2 (MRPC)",
              "new_s1": "New S1 (MRPC-R1)", "new_s2": "New S2 (MRPC-R1)"}, inplace=True)

st.markdown("**MRPC Paraphrase Data Explorer** (Displaying " +
            str(len(df_sel))+" items)")

st.table(data=df_sel)
