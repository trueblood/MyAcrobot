// WebSocket URL (replace with the appropriate server URL)
const serverUrl = "ws://localhost:8089/ws";

/**
 * Initialize WebSocket and set up message handling.
 * @param {function} onMessageCallback - A callback function to handle messages from the WebSocket server.
 * @returns {Promise<WebSocket>} A promise that resolves with the WebSocket instance once connected.
 */
export async function initializeWebSocket(onMessageCallback) {
    return new Promise((resolve, reject) => {
        const socket = new WebSocket(serverUrl);
        console.log("Connecting to WebSocket server at:", serverUrl);

        // Handle successful connection
        socket.onopen = () => {
            console.log("WebSocket connection established. Tim");
           // resolve(socket);

            // Example: Send a prediction request upon connection
            // const exampleState = [0.06528811, 0.99786645, -0.03661686, -0.9993294, 0.02972009, 1.3947774];
            // socket.send(JSON.stringify({ action: "predict", state: exampleState }));
            // Send a test message upon connection
           // socket.send(JSON.stringify({ action: "socket_test" }));

            // Keep the connection alive by sending periodic pings
            // setInterval(() => {
            //     if (socket.readyState === WebSocket.OPEN) {
            //         socket.send(JSON.stringify({ action: "ping" }));
            //         console.log("Sent heartbeat to keep connection alive.");
            //     }
            // }, 30000); // Every 30 seconds
        };

        // Handle connection errors
        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
            reject(error);
        };

        // Handle incoming messages
        socket.onmessage = (message) => {
            console.log("before message recevied from initial connect");
            console.log("WebSocket message received:", message.data); // Logs the raw message data
        };
        
        // Handle incoming messages
        // socket.onmessage = (message) => {
        // //    const data = JSON.parse(message.data); // Parse JSON message
        //     console.log("WebSocket message received:", data);
        //     // if (onMessageCallback) {
        //     //     onMessageCallback(data); // Call the provided callback with the data
        //     // }
        // };

        // Handle connection closure
        socket.onclose = () => {
            console.log("WebSocket connection closed.");
        };
    });
}

// Example WebSocket message handler
export const handleWebSocketMessage = (data) => {
    console.log("Received data Tim:", data);
    if (data.prediction) {
        console.log("Prediction Tim:", data.prediction);
    }
};
