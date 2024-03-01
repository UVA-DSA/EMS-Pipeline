const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');

// Create an HTTP server
const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Socket.IO Server\n');
});

// Attach the Socket.IO server to the HTTP server
const io = socketIo(server);

// Set up a connection event handler for new clients
io.on('connection', (socket) => {
    console.log('A user connected');

    // Set up an event listener for the "message" event
    socket.on('message', (message) => {
        console.log('Received message: ' + message);

        // Broadcast the message to all connected clients, including the sender
        io.emit('message', message);
    });

    // Set up an event listener for the "message" event
    socket.on('byteArray', (message) => {
        console.log('Received bytes!');

                // Assuming byteArrayString is the byte array in string format
        const byteArrayString = message; // Replace with your byte array string

        // Convert the byte array string to a Buffer
        const byteArrayBuffer = Buffer.from(byteArrayString, 'base64'); // You may need to change the encoding if your byte array is in a different format

        // Specify the file path where you want to save the image
        const imagePath = 'image.jpeg'; // You can change the file format as needed (e.g., .jpg, .png)

        // Write the Buffer data to a file to save it as an image
        fs.writeFileSync(imagePath, byteArrayBuffer, 'binary', (err) => {
        if (err) {
            console.error('Error writing image file:', err);
        } else {
            console.log('Image saved successfully.');
            
        }
});


        // // Broadcast the message to all connected clients, including the sender
        // io.emit('message', message);
    });

    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('A user disconnected');
    });
});

// Start the server on port 3000
const port = 5000;
server.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
