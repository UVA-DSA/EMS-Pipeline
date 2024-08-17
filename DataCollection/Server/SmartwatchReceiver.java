import java.io.*;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class SmartwatchReceiver {

    public static void receiveSmartwatchData(String serverIp, int serverPort, BlockingQueue<String[]> fifoQueue, String recordingDir, String smartwatchId, Thread threadStop) {
        try {
            String[] columns = {"sw_epoch_ms", "wrist_position", "sensor_type", "value_X_Axis", "value_Y_Axis", "value_Z_Axis", "seq_num", "server_epoch_ms"};
            String currDate = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("dd-MM-yyyy-HH-mm-ss"));
            String newpath = Paths.get(recordingDir, "smartwatch_data/sw_" + smartwatchId + "/").toString();

            Files.createDirectories(Paths.get(newpath));

            while (true) {
                Socket clientSocket = null;
                boolean connected = false;

                while (!connected) {
                    if (threadStop.isInterrupted()) {
                        break;
                    }
                    try {
                        clientSocket = new Socket(serverIp, serverPort);
                        clientSocket.setSoTimeout(5000);
                        System.out.println("[Smartwatch: Successfully connected to server " + serverIp + ":" + serverPort + "]");
                        connected = true;
                    } catch (SocketTimeoutException e) {
                        System.out.println("[Smartwatch: Connection timeout. Retrying in 5 seconds...]");
                        Thread.sleep(5000);
                    } catch (IOException e) {
                        System.out.println("[Smartwatch: Connection failed: " + e.getMessage() + ". Retrying in 5 seconds...]");
                        Thread.sleep(5000);
                    }
                }

                if (clientSocket != null && connected) {
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter(newpath + "/sw_data.txt", true))) {
                        String message = "Hello, Smart Watch!";
                        while (true) {
                            if (threadStop.isInterrupted()) {
                                clientSocket.close();
                                System.out.println("[Smartwatch: Thread stop set, exiting..]");
                                return;
                            }

                            // Send the message
                            clientSocket.getOutputStream().write(message.getBytes());
                            clientSocket.getOutputStream().flush();

                            // Receive response from the server
                            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                            byte[] buffer = new byte[8192];
                            int bytesRead;
                            while ((bytesRead = clientSocket.getInputStream().read(buffer)) != -1) {
                                byteArrayOutputStream.write(buffer, 0, bytesRead);
                                // Check if we have received all the expected data
                                String partialData = byteArrayOutputStream.toString("UTF-8");
                                if (partialData.endsWith(";")) {
                                    break;
                                }
                            }

                            String receivedMessage = byteArrayOutputStream.toString("UTF-8");
                            System.out.println("[Smartwatch Receiver: Received raw data: " + receivedMessage + "]");

                            String[] dataPoints = receivedMessage.split(";");
                            long currEpochTime = System.currentTimeMillis();

                            for (String dataPoint : dataPoints) {
                                if (!dataPoint.trim().isEmpty()) {
                                    String[] swData = dataPoint.split(",");
                                    
                                    // Check if swData array has the expected length before accessing
                                    if (swData.length >= 7) {
                                        try {
                                            swData[0] = String.valueOf(Long.parseLong(swData[0]));  // sw_epoch_ms
                                            swData[3] = String.valueOf(Double.parseDouble(swData[3]));  // value_X_Axis
                                            swData[4] = String.valueOf(Double.parseDouble(swData[4]));  // value_Y_Axis
                                            swData[5] = String.valueOf(Double.parseDouble(swData[5]));  // value_Z_Axis
                                            swData[6] = String.valueOf(Long.parseLong(swData[7]));  // seq_num

                                            String[] finalSwData = new String[swData.length + 1];
                                            System.arraycopy(swData, 0, finalSwData, 0, swData.length);
                                            finalSwData[finalSwData.length - 1] = String.valueOf(currEpochTime);

                                            writer.write(String.join(",", finalSwData));
                                            writer.newLine();

                                            try {
                                                fifoQueue.put(finalSwData);
                                            } catch (InterruptedException e) {
                                                System.out.println("Error when writing to the FIFO queue: " + e.getMessage());
                                                fifoQueue.clear();
                                            }
                                        } catch (NumberFormatException e) {
                                            System.out.println("[Smartwatch Receiver: Data format error: " + e.getMessage() + "]");
                                        }
                                    } else {
                                        System.out.println("[Smartwatch Receiver: Received data with unexpected format: " + dataPoint + "]");
                                    }
                                }
                            }
                        }
                    } catch (IOException e) {
                        System.out.println("[Error occurred while communicating with server: " + e.getMessage() + "]");
                    }
                }
            }
        } catch (IOException | InterruptedException e) {
            System.out.println("[Smartwatch: Error: " + e.getMessage() + "]");
        }
    }

    public static void main(String[] args) {
        String smartwatch1Ip = "192.168.0.17";
        int smartwatchPort = 7889;
        String smartwatch1Id = "right";
        BlockingQueue<String[]> smartwatch1Q = new LinkedBlockingQueue<>();
        Thread threadStop = Thread.currentThread();

        receiveSmartwatchData(smartwatch1Ip, smartwatchPort, smartwatch1Q, "./test/", smartwatch1Id, threadStop);
    }
}
