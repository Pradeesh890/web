import streamlit as st

def create_direct_download_link(shareable_link):
    file_id = shareable_link.split('/d/')[1].split('/')[0]
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    return direct_link

files = {
    "UDP SERVER JAVA": "https://drive.google.com/file/d/1dcRxuAz_D-bXrN0TPZsS7qS3g738jRVc/view?usp=drive_link",
    "UDP CLIENT JAVA": "https://drive.google.com/file/d/1fzkgsgxUuVg3T3dHmpPvw2q_Si8TDAsp/view?usp=drive_link",
    "TCP SERVER JAVA": "https://drive.google.com/file/d/1tbwK5h7MZGMdnTv2rhxri9rJWn1DCFBK/view?usp=drive_link",
    "TCP CLIENT JAVA": "https://drive.google.com/file/d/19FGPILKKwWj12MBus9uaWQSH4duLryk2/view?usp=drive_link",
    "LINK STATE ROUTING": "https://drive.google.com/file/d/1QfCYWcZpdNYzQoxz8xQ1FjZ8GMPSZx9Y/view?usp=sharing",
    "TCP EMAIL CISCO": "https://drive.google.com/file/d/1uVMUQ3yignyX8_Z6bjGTSf9JzboO5fsL/view?usp=drive_link",
    "STUDY OF PACKET TRACER": "https://drive.google.com/file/d/1nFmOoY_Mu0-eRSn2l_GGBRvjiCPOLRej/view?usp=drive_link",
    "STAR TOPOLOGY CISCO": "https://drive.google.com/file/d/1wMZuS6PyJPB0HrZZscG2xGakal9CbH-l/view?usp=drive_link",
    "RING TOPOLOGY CISCO": "https://drive.google.com/file/d/1DOCUmwJUy6DsH-WCCZdkD66UoujyXtIx/view?usp=drive_link",
    "MESH TOPOLOGY CISCO": "https://drive.google.com/file/d/1QvBmQ1BdINFR205RFP3NE5r1oA7vriZE/view?usp=drive_link",
    "HTTP AND FTP CISCO": "https://drive.google.com/file/d/1gkatOgMo5ld1X6_GnAH8oZ0lLM3zH2l1/view?usp=drive_link",
    "BUS TOPOLOGY CISCO": "https://drive.google.com/file/d/1u4WC6oKL1snGK_VV930J8TJ-xj5bkSMx/view?usp=drive_link"
}

for label, shareable_link in files.items():
    direct_link = create_direct_download_link(shareable_link)
    link_html = f"""
        <a href="{direct_link}" download>
            <h2>{label}</h2>
        </a>
    """
    st.markdown(link_html, unsafe_allow_html=True)
