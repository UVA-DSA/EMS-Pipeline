// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//

#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <string.h>
#include <ctime>
#include <chrono>


int main(int argc, char **argv)
{


    std::string wavfile = "../Audio_Scenarios/2019_Test/";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        printf("Arg %s \n",argv[i]);

        if (arg == "-f" || arg == "--file")
        {
            wavfile += argv[++i];
            printf("Wavefile name %s\n", wavfile.c_str());

        }
        else{
            printf("Wavefile argument failed!\n");
            return -1;
        }
    }

    int fd;
    const char *myfifo = "/tmp/myfifo";
    /* create the FIFO (named pipe) */
    mkfifo(myfifo, 0666);

    SDL_Init(SDL_INIT_AUDIO);

    // load WAV file
    
    SDL_AudioSpec wavSpec;
    Uint32 wavLength;
    Uint8 *wavBuffer;
    
    SDL_LoadWAV(wavfile.c_str(), &wavSpec, &wavBuffer, &wavLength);

    // open audio device
    SDL_AudioDeviceID deviceId = SDL_OpenAudioDevice(NULL, 0, &wavSpec, NULL, 0); 

    // play audio
 
    int success = SDL_QueueAudio(deviceId, wavBuffer, wavLength);
    SDL_PauseAudioDevice(deviceId, 0);

    // keep application running long enough to hear the sound
    
        // Wait for audio to finish playing
  while (SDL_GetQueuedAudioSize(deviceId) > 0)
    {
        // Adjust the sleep duration to check the status more or less frequently
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }


    //  SDL_Delay(60000);

    // clean up
    
    SDL_CloseAudioDevice(deviceId);
    SDL_FreeWAV(wavBuffer);
    SDL_Quit();
    

    return 0;
}
