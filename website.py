import streamlit as st

def create_direct_download_link(shareable_link):
    file_id = shareable_link.split('/d/')[1].split('/')[0]
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    return direct_link

def main():
    st.markdown("## UDP SERVER JAVA")
    udp = """import java.io.*;
import java.net.*;
 class UDPServer
{
   public static void main(String args[]) throws Exception
      {
         DatagramSocket serverSocket = new DatagramSocket(9876);
            byte[] receiveData = new byte[1024];
            byte[] sendData = new byte[1024];
            while(true)
               {
                  DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);
                  serverSocket.receive(receivePacket);
                  String sentence = new String( receivePacket.getData());
                  System.out.println("RECEIVED: " + sentence);
                  InetAddress IPAddress = receivePacket.getAddress();
                  int port = receivePacket.getPort();
                  String capitalizedSentence = sentence.toUpperCase();
                  sendData = capitalizedSentence.getBytes();
                  DatagramPacket sendPacket =
                  new DatagramPacket(sendData, sendData.length, IPAddress, port);
                  serverSocket.send(sendPacket);
               }
      }
}"""
    st.code(udp, language='java')

    st.markdown("UDP CLIENT JAVA")
    udpserver="""import java.io.*;
import java.net.*;
class UDPClient
{
   public static void main(String args[]) throws Exception
   {
      BufferedReader inFromUser =
         new BufferedReader(new InputStreamReader(System.in));
      DatagramSocket clientSocket = new DatagramSocket();
      InetAddress IPAddress = InetAddress.getByName("localhost");
      byte[] sendData = new byte[1024];
      byte[] receiveData = new byte[1024];
      String sentence = inFromUser.readLine();
      sendData = sentence.getBytes();
      DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, IPAddress,   
      9876);
      clientSocket.send(sendPacket);
      DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);
      clientSocket.receive(receivePacket);
      String modifiedSentence = new String(receivePacket.getData());
      System.out.println("FROM SERVER:" + modifiedSentence);
      clientSocket.close();
   }
}
"""
    st.code(udpserver,language='java')

    st.markdown("TCP SERVER JAVA")
    tcpserver="""import java.net.*;
 import java.io.*;
public class TCPServer
{ 
  public static void main(String s[])throws IOException
  {
    //  Initialising the ServerSocket
      ServerSocket sok = new ServerSocket(3128); 
   // Gives the Server Details Machine name, Port number
      System.out.println("Server Started  :"+sok);
   // makes a socket connection to particular client after 
        // which two way communication take place 
     Socket so = sok.accept();
     System.out.println("Client Connected  :"+ so);   
     InputStream in = so.getInputStream(); 
     OutputStream os = so.getOutputStream();
     PrintWriter pr = new PrintWriter(os);
BufferedReader br = new BufferedReader(new  InputStreamReader(in));
BufferedReader br1 = new BufferedReader(new InputStreamReader(System.in));      
while(true)
     {
         System.out.println("Msg  frm client: "+ br.readLine());
 System.out.print("Msg to client: "); 
         pr.println(br1.readLine());
         pr.flush();
     }   }
}    """

    st.code(tcpserver,language='java')

    st.amrkdown("TCP CLIENT JAVA")
    tcpclient="""import java.net.*;
import java.io.*; 
public class TCPClient
{ 
 public static void main(String s[])throws IOException
 { 
  Socket sok = new Socket("localhost",3128);
  InputStream in = sok.getInputStream(); 
  OutputStream ou = sok.getOutputStream();
  PrintWriter pr = new PrintWriter(ou); 
  BufferedReader br1 = new BufferedReader(new InputStreamReader(in));
  BufferedReader br = new BufferedReader(new InputStreamReader(System.in)); 
  while(true)
  { 
   System.out.print("Msg to Server:");
   pr.println(br.readLine()); 
   pr.flush();
   System.out.println("Msg  frm server: "+br1.readLine()); 
  }
 } 
}    

"""
    st.code(tcpclient,language='java')

    files = {
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

if __name__ == '__main__':
    main()
