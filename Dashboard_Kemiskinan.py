import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import json
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# SELECTION CARD
# =========================
st.markdown("""
<style>
.card {
    background-color: #CFECF3;
    padding: 5px;
    border-radius: 10px;
    text-align: center;
    color: black;
    font-weight: bold;
    margin: 5px;
    margin-bottom: 20px;
}

.growth-up{
    color:green;
    font-size:14px;
    font-weight:600;
    margin-top:5px;
}

.growth-down{
    color:red;
    font-size:14px;
    font-weight:600;
    margin-top:5px;
}
            
.card-title{
    font-size:15px;
    opacity:0.9;
}
.card-value{
    font-size:30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# INSIGHT KORELASI
# =========================
st.markdown("""
<style>
.box-penjelasan {
    background-color: #f5f7ff;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    border-left: 5px solid #4a6cf7;
}
</style>
""", unsafe_allow_html=True)

# =========================
# BOX CLUSTER
# =========================

st.markdown("""
<style>

.box-cluster{
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:white;
    font-weight:bold;
    margin-top:2px;
    margin-bottom:150px;
}

.box-merah{
    background-color:#CC3636;
}

.box-orange{
    background-color:#FF9D23;
}

.box-hijau{
    background-color:#367E18;
}

.jumlah{
    font-size:40px;
}

.label{
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ===============
# PROFIL CLUSTER
# ===============
st.markdown("""
<style>
.box-tinggi{
    background-color:#fde2e2;
    border:2px solid #dc3545;
    padding:25px;
    border-radius:12px;
}

.box-sedang{
    background-color:#F8FAB4;
    border:2px solid #FF9D23;
    padding:25px;
    border-radius:12px;
}

.box-rendah{
    background-color:#C7EABB;
    border:2px solid #468432;
    padding:25px;
    border-radius:12px;
}

.row{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    margin-bottom:8px;
    font-size:18px;
}

.row span:first-child{
    text-align:left;
}

.row span:last-child{
    text-align:right;
    font-weight:600;
}

.row.header{
    font-weight:bold;
}

.mean{
    text-align:right;
}

.title{
    font-weight:bold;
    font-size:24px;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# WARNING
# =========================
st.markdown("""
<style>

.warning-tinggi{
    background-color: #FFCDC9;
    border-left:6px solid #d62828;
    padding:18px;
    border-radius:10px;
    margin-top:20px;
    margin-bottom:80px;
    font-size:17px;
}
            
.warning-sedang{
    background-color:#FADA7A;
    border-left:6px solid #FF9D23;
    padding:18px;
    border-radius:10px;
    margin-top:20px;
    margin-bottom:80px;
    font-size:17px;
}

.warning-rendah{
    background-color:#B5E18B;
    border-left:6px solid #468432;
    padding:18px;
    border-radius:10px;
    margin-top:20px;
    margin-bottom:80px;
    font-size:17px;
}

.warning-title{
    font-weight:bold;
    margin-bottom:8px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Hasil_Final.csv")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    data = joblib.load('KNN_Model.joblib')
    return data['model'], data['scaler'], data['features']

model, scaler, features_used = load_model()

# =========================
# LOAD DATA MAP
# =========================
@st.cache_data
def load_map():
    df = pd.read_csv("kabupaten_kota_cluster.csv")

    with open("Kabupaten-Kota_(Provinsi Jawa Timur).geojson") as f:
        geojson = json.load(f)

    return df, geojson

df_map, geojson = load_map()


# =========================
# SESSION STATE PAGE
# =========================
if 'page' not in st.session_state:
    st.session_state.page = "visualisasi"


# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.image(
        "logo_kominfo.png",
    )

    st.markdown("## Menu")

    if st.button("📊 Data Overview", use_container_width=True):
        st.session_state.page = "visualisasi"
        st.rerun()

    if st.button("📉 Analisis Korelasi", use_container_width=True):
        st.session_state.page = "korelasi"
        st.rerun()

    if st.button("🗺️ Peta Cluster Kemiskinan", use_container_width=True):
        st.session_state.page = "cluster"
        st.rerun()

    if st.button("🔍 Prediksi Data", use_container_width=True):
        st.session_state.page = "prediksi"
        st.rerun()

    st.markdown("---")
    st.info("💡 Gunakan halaman prediksi untuk mencoba simulasi cluster kemiskinan.")


# =========================
# HALAMAN VISUALISASI
# =========================
if st.session_state.page == "visualisasi":

    st.title("Dashboard Tingkat Kemiskinan Prov.Jawa Timur")
    st.write("Visualisasi ini menunjukkan tren data setiap varibael di kabupaten/kota jawa timur.")

## Selection Card

    st.subheader("📊 Profil Daerah")

    daerah = st.selectbox(
        "Pilih Kabupaten/Kota",
        df["Kabupaten/Kota Se Jawa Timur"].unique() 
    )

    data_daerah = df[df["Kabupaten/Kota Se Jawa Timur"] == daerah]

    gk = data_daerah["Garis Kemiskinan (Rupiah/Bulan/Kapita)"].values[0]
    p1 = data_daerah["Indeks Kedalaman Kemiskinan (P1)"].values[0]
    p2 = data_daerah["Indeks Keparahan Kemiskinan (P2)"].values[0]
    umk = data_daerah["UMK (Rupiah)"].values[0]
    tpt = data_daerah["TPT (Persen)"].values[0]
    gini = data_daerah["Gini Rasio"].values[0]
    rasio = data_daerah["Rasio UMK & GK"].values[0]
    bansos = data_daerah["Jumlah Penerima Bantuan Sosial"].values[0]

    ## growth rate

    growth_gk = 3.68
    growth_umk = 16.49
    growth_tpt = -6.47
    growth_p1 = -11.9
    growth_p2 = -17.79
    growth_gini = -3.39

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Garis Kemiskinan (Rupiah)</div>
            <div class="card-value">{gk:,.0f}</div>
            <div class="growth-up">
            ▲ {growth_gk:.2f}% dari 2024
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Indeks P1</div>
            <div class="card-value">{p1}</div>
            <div class="growth-down">
            ▼ {abs(growth_p1):.2f}% dari 2024
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Indeks P2</div>
            <div class="card-value">{p2}</div>
            <div class="growth-down">
            ▼ {abs(growth_p2):.2f}% dari 2024
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">UMK (Rupiah)</div>
            <div class="card-value">{umk:,.0f}</div>
            <div class="growth-up">
            ▲ {growth_umk:.2f}% dari 2024
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">TPT(%)</div>
            <div class="card-value">{tpt}</div>
            <div class="growth-down">
            ▼ {abs(growth_tpt):.2f}% dari 2024
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Gini Rasio</div>
            <div class="card-value">{gini}</div>
            <div class="growth-down">
            ▼ {abs(growth_gini):.2f}% dari 2024
        </div>
        """, unsafe_allow_html=True)
   
    col7, col8 = st.columns(2)

    with col7:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Rasio Kesejahteraan (UMK/GK)</div>
            <div class="card-value">{rasio:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Jumlah Bantuan Sosial Kemiskinan Ekstrim</div>
            <div class="card-value">{bansos:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")

# --------------------------------------
# Diagram Batang & Garis Penduduk Miskin
# --------------------------------------
    tahun = ["Mar'21", "Mar'22", "Mar'23", "Mar'24", "Mar'25"]

    jumlah_orang = [4.573, 4.181, 4.188, 3.983, 3.876]
    persen = [11.40, 10.38, 10.35, 9.79, 9.50]

    # gambar
    fig, ax1 = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#f2f2f2")
    ax1.set_facecolor("#f2f2f2")

### Diagram batang
    bars = ax1.bar(
        tahun,
        jumlah_orang,
        color="#014F6E",
        edgecolor="black",
        width=0.50
    )

    # Label di dalam batang
    for bar, value in zip(bars, jumlah_orang):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            0.15,
            f"{value:.3f}",
            ha='center',
            color='white',
            fontsize=11,
            fontweight='bold'
        )

    # Nama sumbu kiri
    ax1.set_ylabel("Jumlah Penduduk Miskin (Juta Orang)",
                fontsize=11,
                fontweight='bold')

    # Nama sumbu bawah
    ax1.set_xlabel("Periode",
                fontsize=11,
                fontweight='bold')

### Diagram garis
    ax2 = ax1.twinx()

    ax2.plot(
        tahun,
        persen,
        color="#E66A2C",
        marker='o',
        linewidth=4,
        markersize=15
    )

    # Supaya garis ada di atas batang
    ax2.set_ylim(8.5, 12)

    # Label persen di atas titik
    for x, y in zip(tahun, persen):
        ax2.text(
            x,
            y + 0.23,
            f"{str(y).replace('.', ',')}%",
            ha='center',
            fontsize=11,
            fontweight='bold',
            color='white'
        )

    # Nama sumbu kanan
    ax2.set_ylabel("Persentase Kemiskinan (%)",
                fontsize=11,
                fontweight='bold')

    # Gaya
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Judul
    plt.title(
        "Perkembangan Penduduk Miskin Jawa Timur Tahun 2021-2025",
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    st.pyplot(fig)

    st.write("")
    st.write("")

# -------------------------------
# Diagram Batang & Garis P1 & P2
# -------------------------------
    tahun = ["Mar'21", "Mar'22", "Mar'23", "Mar'24", "Mar'25"]

    total_p1 = [1.841, 1.618, 1.630, 1.478, 1.414]
    total_p2 = [0.429, 0.377, 0.366, 0.313, 0.294]

    # gambar
    fig, ax3 = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#f2f2f2")
    ax3.set_facecolor("#f2f2f2")

### Diagram batang
    bars = ax3.bar(
        tahun,
        total_p1,
        color="#014F6E",
        edgecolor="black",
        width=0.50
    )

    # Label di dalam batang
    for bar, value in zip(bars, total_p1):
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            0.15,
            f"{value:.3f}",
            ha='center',
            color='white',
            fontsize=11,
            fontweight='bold'
        )

    # Nama sumbu kiri
    ax3.set_ylabel("Indeks P1",
                fontsize=11,
                fontweight='bold')

    # Nama sumbu bawah
    ax3.set_xlabel("Periode",
                fontsize=11,
                fontweight='bold')

### Diagram garis
    ax4 = ax3.twinx()

    ax4.plot(
        tahun,
        total_p2,
        color="#E66A2C",
        marker='o',
        linewidth=4,
        markersize=15
    )

    # Supaya garis ada di atas batang
    ax4.set_ylim(0, 1)

    # Label persen di atas titik
    for x, y in zip(tahun, total_p2):
        ax4.text(
            x,
            y + 0.1,
            f"{str(y).replace('.', ',')}%",
            ha='center',
            fontsize=11,
            fontweight='bold',
            color='white'
        )

    # Nama sumbu kanan
    ax4.set_ylabel("Indeks P2",
                fontsize=11,
                fontweight='bold')

    # Gaya
    ax3.spines['top'].set_visible(False)
    ax4.spines['top'].set_visible(False)

    # Judul
    plt.title(
        "Perkembangan Indeks P1 dan P2 Jawa Timur Tahun 2021-2025",
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    st.pyplot(fig)

    st.write("")
    st.write("")
    st.write("")
    st.write("")

# ---------------------------------------
# Diagram Perbandingan P1 dan P2 Kab/Kota
# ---------------------------------------
    st.title("Perbandingan P1 dan P2 Setiap Kota/Kabupaten Jawa Timur")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        df["Kabupaten/Kota Se Jawa Timur"],
        df["Indeks Kedalaman Kemiskinan (P1)"],
        marker='o',
        label="P1"
    )

    ax.plot(
        df["Kabupaten/Kota Se Jawa Timur"],
        df["Indeks Keparahan Kemiskinan (P2)"],
        marker='o',
        label="P2"
    )

    ax.set_title("Perbandingan P1 dan P2")
    ax.legend()
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

### Diagram Batang P2
    # Ambil Top 5 dan Bottom 5
    top5_p1 = df.nlargest(5, "Indeks Kedalaman Kemiskinan (P1)")
    bottom5_p1 = df.nsmallest(5, "Indeks Kedalaman Kemiskinan (P1)")

    # Buat diagram batang
    fig_top = px.bar(
        top5_p1,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Kedalaman Kemiskinan (P1)",
        orientation="h",
        title="Top 5 Nilai P1 Tertinggi",
        color_discrete_sequence=["red"]
    )

    fig_bottom = px.bar(
        bottom5_p1,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Kedalaman Kemiskinan (P1)",
        orientation="h",
        title="Top 5 Nilai P1 Terendah",
        color_discrete_sequence=["green"]
    )
    
    # Tampilkan berdampingan
    col1, col2 = st.columns(2)
    with col1:
     st.plotly_chart(fig_top, use_container_width=True)
    with col2:
     st.plotly_chart(fig_bottom, use_container_width=True)

### Diagram Batang P2
    # Ambil Top 5 dan Bottom 5
    top5_p2 = df.nlargest(5, "Indeks Keparahan Kemiskinan (P2)")
    bottom5_p2 = df.nsmallest(5, "Indeks Keparahan Kemiskinan (P2)")

    # Buat diagram batang
    fig_top = px.bar(
        top5_p2,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Keparahan Kemiskinan (P2)",
        orientation="h",
        title="Top 5 Nilai P2 Tertinggi",
        color_discrete_sequence=["red"]
    )

    fig_bottom = px.bar(
        bottom5_p2,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Keparahan Kemiskinan (P2)",
        orientation="h",
        title="Top 5 Nilai P2 Terendah",
        color_discrete_sequence=["green"]
    )
    
    # Tampilkan berdampingan
    col3, col4 = st.columns(2)
    with col3:
     st.plotly_chart(fig_top, use_container_width=True)
    with col4:
     st.plotly_chart(fig_bottom, use_container_width=True)

# =========================
# HALAMAN KORELASI
# =========================
elif st.session_state.page == "korelasi":
    
    st.title("Analisis Korelasi Variabel")
    st.write("Visualisasi hubungan antara indikator kemiskinan.")

    var1 = 'Indeks Keparahan Kemiskinan (P2)'
    var2 = 'Indeks Kedalaman Kemiskinan (P1)'

    fig, ax = plt.subplots(figsize=(8,6))

    sns.regplot(
        data=df,
        x=var1,
        y=var2,
        ci=None,
        scatter_kws={'color':'blue'},
        line_kws={'color':'red','linestyle':'--'},
        ax=ax
    )

    ax.set_title("Korelasi Positif antara P1 dan P2")
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.grid(True)

    st.pyplot(fig)

    # Hitung korelasi
    correlation = df[var1].corr(df[var2])

    st.metric(
        label="Koefisien Korelasi",
        value=f"{correlation:.2f}%"
    )

### Insight Korelasi
    st.markdown("""
    <div class="box-penjelasan">
    Korelasi antara indeks kedalaman kemiskinan (P1) dan indeks keparahan kemiskinan (P2) sebesar 95%. 
    Hal ini menunjukkan bahwa pada 36 kabupaten/kota, semakin jauh rata-rata pengeluaran penduduk miskin 
    dari garis kemiskinan, maka ketimpangan pengeluaran antar penduduk miskin juga cenderung semakin tinggi.
    Namun pada sekitar 5% wilayah (2 kabupaten/kota), pola hubungan tersebut tidak terlihat, 
    seperti pada Kabupaten Sidoarjo dan Kota Blitar.
    </div>
    """, unsafe_allow_html=True)

# =========================
# HALAMAN CLUSTER
# =========================
elif st.session_state.page == "cluster":

    st.title("Peta Cluster Kemiskinan Jawa Timur 2025")
    st.write("Visualisasi peta ini menunjukkan pengelompokan kabupaten/kota berdasarkan tingkat kemiskinan.")

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="Kabupaten/Kota Se Jawa Timur",
        featureidkey="properties.NAME_2",
        color="Cluster",

        color_discrete_map={
            'Daerah dengan Tingkat Kemiskinan Tinggi': "red",
            'Daerah dengan Tingkat Kemiskinan Sedang': "orange",
            'Daerah dengan Tingkat Kemiskinan Rendah': "green",
        },

        hover_name="Kabupaten/Kota Se Jawa Timur"
    )

    fig.update_geos(fitbounds="locations", visible=False)

    st.plotly_chart(fig, use_container_width=True)

### Donat Chart
    st.title("Visualisasi Hasil Cluster")

    total_daerah = df["Kabupaten/Kota Se Jawa Timur"].nunique()
    fig = px.pie(
        df,
        names="Cluster",
        hole=0.5,
        color="Cluster",
        color_discrete_map={
            'Daerah dengan Tingkat Kemiskinan Tinggi': "red",
            'Daerah dengan Tingkat Kemiskinan Sedang': "orange",
            'Daerah dengan Tingkat Kemiskinan Rendah': "green",
            }
    )

    # teks tengah
    fig.update_layout(
        annotations=[dict(
            text=f"🏙️{total_daerah}<br>Kab/Kota",
            x=0.5,
            y=0.5,
            font=dict(
                size=25,
                color="#2c3e50"
            ),
            showarrow=False
        )]
    )

    # teks persen
    fig.update_traces(
    textinfo="percent", 
    textposition="inside",
    insidetextorientation="horizontal", 
    textfont=dict(
        size=25,        
        color="white"  
        )
    )

    st.plotly_chart(fig)

### Kotak total cluster
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="box-cluster box-merah">
            <div class="jumlah">11</div>
            <div class="label">Daerah Tingkat Kemiskinan Tinggi</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="box-cluster box-orange">
            <div class="jumlah">6</div>
            <div class="label">Daerah Tingkat Kemiskinan Sedang</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="box-cluster box-hijau">
            <div class="jumlah">21</div>
            <div class="label">Daerah Tingkat Kemiskinan Rendah</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Keterangan Cluster Tinggi
# -------------------------
    st.title("Profil Daerah Cluster")
    df_tinggi = df[df["Cluster"] == "Daerah dengan Tingkat Kemiskinan Tinggi"]

    rata_gk = df_tinggi["Garis Kemiskinan (Rupiah/Bulan/Kapita)"].mean()
    rata_p1 = df_tinggi["Indeks Kedalaman Kemiskinan (P1)"].mean()
    rata_p2 = df_tinggi["Indeks Keparahan Kemiskinan (P2)"].mean()
    rata_gini = df_tinggi["Gini Rasio"].mean()
    rata_umk = df_tinggi["UMK (Rupiah)"].mean()
    rata_tpt = df_tinggi["TPT (Persen)"].mean()
    rata_rasio = df_tinggi["Rasio UMK & GK"].mean()

    st.markdown(f"""
    <div class="box-tinggi">

    <div class="title"> Profil Daerah Prioritas Tinggi</div>

    <div class="row header">
    <span></span>
    <span class="mean">Mean</span>
    </div>        
        
    <div class="row">
    <span><span style="color:green">▼</span> Garis Kemiskinan Cukup Rendah</span>
    <span>{rata_gk:,.0f}<span>
    </div>

    <div class="row">
    <span><span style="color:red">▲</span> Tingkat Kedalaman Kemiskinan Relatif Tinggi</span>
    <span>{rata_p1:.2f}</span>
    </div>

    <div class="row">
    <span><span style="color:red">▲</span> Tingkat Keparahan Kemiskinan Relatif Tinggi</span>
    <span>{rata_p2:.3f}</span>
    </div>

    <div class="row">
    <span><span style="color:red">▼</span> UMK Sangat Rendah</span>
    <span>{rata_umk:,.0f}</span>
    </div>

    <div class="row">
    <span><span style="color:green">▼</span> TPT Sangat Rendah</span>
    <span>{rata_tpt:.2f}%</span>
    </div>

    <div class="row">
    <span><span style="color:green">▼</span> Gini Rasio Sangat Rendah</span>
    <span>{rata_gini:.2f}%</span>
    </div>

    <div class="row">
    <span><span style="color:red">▼</span> Rasio Kesejahteraan Cukup Buruk</span>
    <span>{rata_rasio:.2f}</span>
    </div>

    </div>
    """, unsafe_allow_html=True)

### warning

    st.markdown(f"""
    <div class="warning-tinggi">

    <div class="warning-title">⚠️ Catatan Penting</div>

    Wilayah dengan prioritas tinggi menunjukkan bahwa meskipun beberapa indikator
    seperti TPT dan Gini Rasio tergolong rendah, daya beli masyarakat masih
    relatif lemah karena nilai UMK yang sangat mendekati Garis Kemiskinan.
    Kondisi ini menunjukkan perlunya intervensi kebijakan untuk meningkatkan
    kesejahteraan masyarakat.

    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Keterangan Cluster Sedang
# -------------------------

    df_sedang = df[df["Cluster"] == "Daerah dengan Tingkat Kemiskinan Sedang"]

    rata_gk = df_sedang["Garis Kemiskinan (Rupiah/Bulan/Kapita)"].mean()
    rata_p1 = df_sedang["Indeks Kedalaman Kemiskinan (P1)"].mean()
    rata_p2 = df_sedang["Indeks Keparahan Kemiskinan (P2)"].mean()
    rata_gini = df_sedang["Gini Rasio"].mean()
    rata_umk = df_sedang["UMK (Rupiah)"].mean()
    rata_tpt = df_sedang["TPT (Persen)"].mean()
    rata_rasio = df_sedang["Rasio UMK & GK"].mean()

    st.markdown(f"""
    <div class="box-sedang">

    <div class="title"> Profil Daerah Prioritas Sedang</div>

    <div class="row header">
    <span></span>
    <span class="mean">Mean</span>
    </div>        
        
    <div class="row">
    <span><span style="color:red">▲</span> Garis Kemiskinan Tinggi</span>
    <span>{rata_gk:,.0f}<span>
    </div>

    <div class="row">
    <span><span style="color:orange">▲</span> Tingkat Kedalaman Kemiskinan Relatif Sedang</span>
    <span>{rata_p1:.2f}</span>
    </div>

    <div class="row">
    <span><span style="color:orange">▲</span> Tingkat Keparahan Kemiskinan Relatif Sedang</span>
    <span>{rata_p2:.3f}</span>
    </div>

    <div class="row">
    <span><span style="color:green">▲</span> UMK Tinggi</span>
    <span>{rata_umk:,.0f}</span>
    </div>

    <div class="row">
    <span><span style="color:red">▲</span> TPT Tinggi</span>
    <span>{rata_tpt:.2f}%</span>
    </div>

    <div class="row">
    <span><span style="color:red">▲</span> Gini Rasio Tinggi</span>
    <span>{rata_gini:.2f}%</span>
    </div>

    <div class="row">
    <span><span style="color:green">▲</span> Rasio Kesejahteraan Sangat Baik</span>
    <span>{rata_rasio:.2f}</span>
    </div>

    </div>
    """, unsafe_allow_html=True)

### warning

    st.markdown(f"""
    <div class="warning-sedang">

    <div class="warning-title">⚠️ Catatan Penting</div>

    Wilayah dengan prioritas sedang menunjukkan bahwa meskipun beberapa indikator
    seperti TPT dan Gini Rasio tergolong rendah, daya beli masyarakat masih
    relatif lemah karena nilai UMK yang sangat mendekati Garis Kemiskinan.
    Kondisi ini menunjukkan perlunya intervensi kebijakan untuk meningkatkan
    kesejahteraan masyarakat.

    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Keterangan Cluster Rendah
# -------------------------

    df_rendah = df[df["Cluster"] == "Daerah dengan Tingkat Kemiskinan Rendah"]

    rata_gk = df_rendah["Garis Kemiskinan (Rupiah/Bulan/Kapita)"].mean()
    rata_p1 = df_rendah["Indeks Kedalaman Kemiskinan (P1)"].mean()
    rata_p2 = df_rendah["Indeks Keparahan Kemiskinan (P2)"].mean()
    rata_gini = df_rendah["Gini Rasio"].mean()
    rata_umk = df_rendah["UMK (Rupiah)"].mean()
    rata_tpt = df_rendah["TPT (Persen)"].mean()
    rata_rasio = df_rendah["Rasio UMK & GK"].mean()

    st.markdown(f"""
    <div class="box-rendah">

    <div class="title"> Profil Daerah Prioritas Sedang</div>

    <div class="row header">
    <span></span>
    <span class="mean">Mean</span>
    </div>        
        
    <div class="row">
    <span><span style="color:orange">▲</span> Garis Kemiskinan Sedang</span>
    <span>{rata_gk:,.0f}<span>
    </div>

    <div class="row">
    <span><span style="color:green">▼</span> Tingkat Kedalaman Kemiskinan Relatif Rendah</span>
    <span>{rata_p1:.2f}</span>
    </div>

    <div class="row">
    <span><span style="color:green">▼</span> Tingkat Keparahan Kemiskinan Relatif Rendah</span>
    <span>{rata_p2:.3f}</span>
    </div>

    <div class="row">
    <span><span style="color:red">▼</span> UMK Rendah</span>
    <span>{rata_umk:,.0f}</span>
    </div>

    <div class="row">
    <span><span style="color:green">▲</span> TPT Tinggi</span>
    <span>{rata_tpt:.2f}%</span>
    </div>

    <div class="row">
    <span><span style="color:orange">▲</span> Gini Rasio Sedang</span>
    <span>{rata_gini:.2f}%</span>
    </div>

    <div class="row">
    <span><span style="color:red">▲</span> Rasio Kesejahteraan Cukup Buruk</span>
    <span>{rata_rasio:.2f}</span>
    </div>

    </div>
    """, unsafe_allow_html=True)

### warning

    st.markdown(f"""
    <div class="warning-rendah">

    <div class="warning-title">⚠️ Catatan Penting</div>

    Wilayah dengan prioritas sedang menunjukkan bahwa meskipun beberapa indikator
    seperti TPT dan Gini Rasio tergolong rendah, daya beli masyarakat masih
    relatif lemah karena nilai UMK yang sangat mendekati Garis Kemiskinan.
    Kondisi ini menunjukkan perlunya intervensi kebijakan untuk meningkatkan
    kesejahteraan masyarakat.

    </div>
    """, unsafe_allow_html=True)

# =========================
# HALAMAN PREDIKSI
# =========================
elif st.session_state.page == "prediksi":

    st.title("Prediksi Cluster Kemiskinan")
    st.write("Masukkan data indikator untuk memprediksi cluster kemiskinan.")

    # INPUT USER
    garis_kemiskinan = st.number_input("Garis Kemiskinan", value=550000.0)
    umk = st.number_input("UMK", value=2500000.0)
    tpt = st.number_input("TPT (%)", value=4.5)
    p1 = st.number_input("Indeks Kedalaman Kemiskinan (P1)", value=0.7)
    p2 = st.number_input("Indeks Keparahan Kemiskinan (P2)", value=1.2)
    gini_rasio = st.number_input("Gini Rasio", value=0.324)
    rasio_umk_gk = st.number_input("Rasio UMK / Garis Kemiskinan", value=2.0)

    st.markdown("---")

    # =========================
    # PREDIKSI
    # =========================
    if st.button("🔍 Prediksi Sekarang"):

        input_data = pd.DataFrame([[
            garis_kemiskinan,
            umk,
            tpt,
            p1,
            p2,
            gini_rasio,
            rasio_umk_gk
        ]], columns=features_used)

        # normalisasi
        input_scaled = scaler.transform(input_data)

        # prediksi
        prediction = model.predict(input_scaled)[0]

        st.success(f"Hasil Prediksi Cluster: **{prediction}**")