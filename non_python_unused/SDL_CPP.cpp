#include "SDL.h"
#include <iostream>
#include <cstring> // Add this line for the cstring library

#ifdef _WIN32
    #include <winsock2.h>
    #include <Ws2tcpip.h>
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

#ifdef _WIN32
WSADATA wsaData;
if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    std::cerr << "WSAStartup failed.\n";
    exit(1);
}
#endif

int sock;
struct sockaddr_in addr;


void AudioCallback(void* userdata, Uint8* stream, int len) {
    struct sockaddr_in cliAddr;
    socklen_t cliAddrLen = sizeof(cliAddr);

    std::cerr << "Length of stream: " << len << "\n";
    // Attempt to receive data from the UDP socket
    int received = recvfrom(sock, (char*)stream, len, 0, (struct sockaddr *)&cliAddr, &cliAddrLen);
    std::cerr << "Received " << received << " bytes\n";
    if (received < 0) {
        // If nothing was received, fill the buffer with silence
        SDL_memset(stream, 0, len);
    }
    // If received data is less than 'len', you might want to fill the rest with silence
}


int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Setup UDP socket...

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Error opening socket\n";
        exit(1);
    }

    memset((char *)&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(8888);  // Use your desired port

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cerr << "Bind failed\n";
        exit(1);
    }

        std::cerr << "Hello" << std::endl;

    // Specify desired audio specs
    SDL_AudioSpec desiredSpec;
    SDL_zero(desiredSpec);
    desiredSpec.freq = 16000; // Sample rate
    desiredSpec.format = AUDIO_S16SYS; // Audio format
    desiredSpec.channels = 1; // Mono audio
    desiredSpec.samples = 512; // Buffer size
    desiredSpec.callback = AudioCallback; // Callback function

    // Open audio device
    SDL_AudioSpec obtainedSpec;
    SDL_AudioDeviceID audioDevice = SDL_OpenAudioDevice(NULL, 0, &desiredSpec, &obtainedSpec, SDL_AUDIO_ALLOW_FORMAT_CHANGE);
    if (audioDevice == 0) {
        std::cerr << "Failed to open audio: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    // Start playing audio
    SDL_PauseAudioDevice(audioDevice, 0);

    // Main loop
    bool running = true;
    SDL_Event e;
    while (running) {
        // Process SDL events
        while (SDL_PollEvent(&e) != 0) {
            // User requests quit
            if (e.type == SDL_QUIT) {
                running = false;
            }
        }

        // Perform any other per-frame logic here

        // In an audio-focused application, there may not be much else to do in the main loop
        // since audio processing is handled in the callback
    }

    // Cleanup
    close(sock); // Close the UDP socket
    SDL_CloseAudioDevice(audioDevice);
    SDL_Quit();

    // Clean up networking if needed, especially on Windows
    #ifdef _WIN32
    WSACleanup();
    #endif

    return 0;
}