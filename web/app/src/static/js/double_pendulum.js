// https://chatgpt.com/c/6766dfef-3500-8000-9c2a-2bdd33925e1c
const trails = {
    lowerArm: [],
    upperArm: [],
};

let currentControl = 'lowerArm'; // Track which link is currently controlled

// You can also track alignment events
let alignmentCount = 0;
let lastAlignmentTime = 0;

let level = 1; // Initialize the level
let randomZone1 = ''; // Random zone for the upper arm
let randomZone2 = ''; // Random zone for the lower arm
let randomZone3 = ''; // Random zone for the upper arm
let randomZone4 = ''; // Random zone for the lower arm
let score = 0; // Initialize the score
let length = 100;
let width = 25;
let numberOfLinks = 2;
let playerDifficultyLevel = 'easy';
let playerName = "Guest";
let airFriction = 0;
const links = [];


// Function to pick a random zone
function pickRandomZone() {
    const zones = ['Top-Left Quadrant', 'Top-Right Quadrant', 'Bottom-Left Quadrant', 'Bottom-Right Quadrant'];
    return zones[Math.floor(Math.random() * zones.length)];
}

// Function to pick a zone adjacent to the given zone
function pickAdjacentZone(currentZone) {
    const adjacentZones = {
        'Top-Left Quadrant': ['Top-Right Quadrant', 'Bottom-Left Quadrant'],
        'Top-Right Quadrant': ['Top-Left Quadrant', 'Bottom-Right Quadrant'],
        'Bottom-Left Quadrant': ['Top-Left Quadrant', 'Bottom-Right Quadrant'],
        'Bottom-Right Quadrant': ['Top-Right Quadrant', 'Bottom-Left Quadrant'],
    };
    const possibleZones = adjacentZones[currentZone];
    return possibleZones[Math.floor(Math.random() * possibleZones.length)];
}

// Function to determine random zones based on the level
// function determineZonesForLevel() {
//     if (level === 1) {
//         // Level 1: Both zones are the same
//         const zone = pickRandomZone();
//         randomZone1 = zone;
//         randomZone2 = zone;
//     } else if (level === 2) {
//         // Level 2: Zones are adjacent
//         randomZone1 = pickRandomZone();
//         randomZone2 = pickAdjacentZone(randomZone1);
//     } else if (level >= 3) {
//         // Level 3: Zones are random
//         randomZone1 = pickRandomZone();
//         randomZone2 = pickRandomZone();
//     }

//     updateRandomZone1(randomZone1);
//     updateRandomZone2(randomZone2);
// }

function determineZonesForLevel(difficulty) {
    // Reset zones
    randomZone1 = '';
    randomZone2 = '';
    randomZone3 = '';
    randomZone4 = '';

    if (difficulty === 'easy') {
        // Easy: 1 zone required
        const zone = pickRandomZone();
        randomZone1 = zone;
        //randomZone2 = zone;
    } else if (difficulty === 'medium') {
        // Medium: 2 zones required
        randomZone1 = pickRandomZone();
        randomZone2 = pickAdjacentZone(randomZone1);
    } else if (difficulty === 'hard') {
        // Hard: 3 zones required
        randomZone1 = pickRandomZone();
        randomZone2 = pickRandomZone();
        while (randomZone2 === randomZone1) {
            randomZone2 = pickRandomZone(); // Ensure zones are different
        }
        randomZone3 = pickRandomZone();
        while (randomZone3 === randomZone1 || randomZone3 === randomZone2) {
            randomZone3 = pickRandomZone(); // Ensure zones are unique
        }
    } else if (difficulty === 'expert') {
        // Expert: 4 zones required
        const usedZones = [];
        while (usedZones.length < 4) {
            const zone = pickRandomZone();
            if (!usedZones.includes(zone)) {
                usedZones.push(zone);
            }
        }
        randomZone1 = usedZones[0];
        randomZone2 = usedZones[1];
        randomZone3 = usedZones[2];
        randomZone4 = usedZones[3];
    }

    // Update the UI with the determined zones
    updateRandomZone1(randomZone1);
    updateRandomZone2(randomZone2);
    updateRandomZone3(randomZone3);
    updateRandomZone4(randomZone4);

    //  console.log(`Difficulty: ${difficulty}`);
    //    console.log(`Zones: ${randomZone1}, ${randomZone2}, ${randomZone3}, ${randomZone4}`);
}

// Functions to update HTML elements
function updateRandomZone1(value) {
    randomZone1 = value;
    const randomZone1Display = document.getElementById('randomZone1Display');
    if (randomZone1Display) {
        randomZone1Display.textContent = `${value}`;
    }
}

function updateRandomZone2(value) {
    randomZone2 = value;
    const randomZone2Display = document.getElementById('randomZone2Display');
    if (randomZone2Display) {
        randomZone2Display.textContent = `${value}`;
    }
}

function updateRandomZone3(value) {
    randomZone3 = value;
    const randomZone3Display = document.getElementById('randomZone3Display');
    if (randomZone3Display) {
        randomZone3Display.textContent = `${value}`;
    }
}

function updateRandomZone4(value) {
    randomZone4 = value;
    const randomZone4Display = document.getElementById('randomZone4Display');
    if (randomZone4Display) {
        randomZone4Display.textContent = `${value}`;
    }
}

function updateScore(value) {
    score = value;
    const scoreDisplay = document.getElementById('scoreDisplay');
    if (scoreDisplay) {
        scoreDisplay.textContent = `${value}`;
    }
}

function updateLevel(value) {
    level = value;
    const levelDisplay = document.getElementById('levelDisplay');
    if (levelDisplay) {
        levelDisplay.textContent = `Level: ${value}`;
    }
}

function removeChain(engine, chainComposite) {
    // Remove all constraints in the chain composite
    chainComposite.constraints.forEach(constraint => {
        Matter.World.remove(engine.world, constraint);
    });

    // Clear the constraints array
    chainComposite.constraints = [];
}

function trackAlignment(upperArm, lowerArm) {
    const currentTime = Date.now();
    const alignmentInfo = checkPendulumsAlignment(upperArm, lowerArm);

    if (alignmentInfo.isAligned) {
        // Only count as new alignment if more than 500ms has passed
        if (currentTime - lastAlignmentTime > 500) {
            alignmentCount++;
            lastAlignmentTime = currentTime;

            // Store alignment data
            const alignmentData = {
                timestamp: currentTime,
                count: alignmentCount,
                upperArmAngle: alignmentInfo.upperDegrees,
                lowerArmAngle: alignmentInfo.lowerDegrees,
                upperArmPosition: { x: upperArm.position.x, y: upperArm.position.y },
                lowerArmPosition: { x: lowerArm.position.x, y: lowerArm.position.y }
            };

            //  console.log('Alignment detected:', alignmentData);

            // Update UI with alignment count
            const countDisplay = document.getElementById('alignmentCountDisplay');
            countDisplay.textContent = `Total Alignments: ${alignmentCount}`;
        }
    }
}

// Add a function to handle pausing the simulation for a specific duration
function pauseSimulation(duration) {
    return new Promise((resolve) => setTimeout(resolve, duration));
}

function checkPendulumsAlignment(upperArm, lowerArm) {
    // Get the angles of both arms in radians
    const upperAngle = upperArm.angle % (2 * Math.PI);
    const lowerAngle = lowerArm.angle % (2 * Math.PI);

    // Convert to degrees for easier understanding
    const upperDegrees = (upperAngle * 180 / Math.PI);
    const lowerDegrees = (lowerAngle * 180 / Math.PI);

    // Define tolerance for "straightness" (in degrees)
    const tolerance = 30; // adjust this value as needed

    // Check if angles are aligned (either both vertical or both horizontal)
    const angleDifference = Math.abs(upperDegrees - lowerDegrees);

    // They are aligned if their angle difference is near 0° or 180°
    const isAligned = (
        angleDifference <= tolerance ||
        Math.abs(angleDifference - 180) <= tolerance
    );

    return {
        isAligned: isAligned,
        upperDegrees: upperDegrees,
        lowerDegrees: lowerDegrees,
        angleDifference: angleDifference
    };
}

function getBestActionFromModel(parsedValue) {
    // Safely navigate to best_action
    if (parsedValue && parsedValue.length > 0 && parsedValue[0]?.data?.best_action !== undefined) {
        return parsedValue[0].data.best_action;
    }
    // console.error("Invalid input structure: Unable to retrieve best_action.");
    return null; // Return null if best_action is not found
}

document.addEventListener('DOMContentLoaded', function () {
    //  console.log("in event listener");
    const hiddenInput = document.getElementById('savedMessagesHidden');
    // const hiddenInput = document.getElementById('hdnPendulumState');
    // console.log('Raw hidden input value:', hiddenInput.value);

    if (hiddenInput) {
        // Listen for changes using multiple approaches to ensure we catch all updates
        hiddenInput.addEventListener('change', function () {
            //      console.log('Value changed from event listner tim:', this.value);
            try {
                const parsedValue = JSON.parse(this.value);
                //      console.log('Parsed value:', parsedValue);
            } catch (error) {
                console.error('Error parsing value:', error);
            }
        });

        // Using MutationObserver to catch programmatic changes
        const observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'value') {
                    //    console.log('Value updated here:', hiddenInput.value);


                    try {
                        const parsedValue = JSON.parse(hiddenInput.value);
                        // console.log('Parsed updated value:', parsedValue);
                    } catch (error) {
                        console.error('Error parsing updated value:', error);
                    }
                }
            });
        });

        observer.observe(hiddenInput, {
            attributes: true
        });
    } else {
        console.error('Hidden input element not found');
    }
});

function getObservationFromPendulum(lowerArm, upperArm) {
    // Extract angles and angular velocities
    const theta1 = lowerArm.angle; // Lower arm angle
    const theta2 = upperArm.angle; // Upper arm angle
    const angularVelocityTheta1 = lowerArm.angularVelocity;
    const angularVelocityTheta2 = upperArm.angularVelocity;

    // Create the observation array
    const observation = [
        Math.cos(theta1), Math.sin(theta1), // Cosine and sine of θ1
        Math.cos(theta2), Math.sin(theta2), // Cosine and sine of θ2
        angularVelocityTheta1, angularVelocityTheta2 // Angular velocities
    ];

    return observation;
}

function logPendulumState(lowerArm, upperArm) {
    console.log("Next group");
    console.log("Lower Arm:");
    console.log(`- Position: x=${lowerArm.position.x.toFixed(2)}, y=${lowerArm.position.y.toFixed(2)}`);
    console.log(`- Velocity: x=${lowerArm.velocity.x.toFixed(2)}, y=${lowerArm.velocity.y.toFixed(2)}`);
    console.log(`- Angle: ${lowerArm.angle.toFixed(2)} radians`);
    console.log(`- Angular Velocity: ${lowerArm.angularVelocity.toFixed(2)}`);

    console.log("Upper Arm:");
    console.log(`- Position: x=${upperArm.position.x.toFixed(2)}, y=${upperArm.position.y.toFixed(2)}`);
    console.log(`- Velocity: x=${upperArm.velocity.x.toFixed(2)}, y=${upperArm.velocity.y.toFixed(2)}`);
    console.log(`- Angle: ${upperArm.angle.toFixed(2)} radians`);
    console.log(`- Angular Velocity: ${upperArm.angularVelocity.toFixed(2)}`);
}

// Function to calculate the y-coordinate of the pendulum's endpoint
function calculateYEndpoint(lowerArm, upperArm, linkLength1, linkLength2) {
    // Calculate theta1 and theta2 based on the rotation of each link
    const theta1 = lowerArm.angle;  // Angle of the first link
    const theta2 = upperArm.angle;  // Angle of the second link

    // Calculate the y position of the endpoint
    const y = -linkLength1 * Math.cos(theta1) - linkLength2 * Math.cos(theta1 + theta2);

    return y;
}

// Function to add touch controls
// function addTouchControl(pendulum, canvas, engine, chainComposite) {
//     const lowerArm = pendulum.bodies[1];
//     const upperArm = pendulum.bodies[0];
//     let isSwitching = false; // Prevent rapid toggling
//     let lastTapTime = 0; // For detecting double-tap

//     // Add touchstart and touchmove event listeners to the canvas
//     canvas.addEventListener('touchstart', handleTouch);
//     canvas.addEventListener('touchmove', handleTouch);
//     canvas.addEventListener('touchend', handleTouchEnd);

//     function handleTouch(event) {
//         event.preventDefault(); // Prevent default touch behavior (e.g., scrolling)

//         // Detect multi-touch
//         const touches = event.touches;
//         const touchCount = touches.length;
//         const currentTime = new Date().getTime();
//         const timeDifference = currentTime - lastTapTime;

//         if (touchCount === 2 && !isSwitching) {
//             // Two-finger touch switches control between links
//             isSwitching = true;
//             switchControl();
//         } else if (touchCount === 1 && timeDifference < 300) {
//             // Double-tap releases the pendulum links
//             removeChain(engine, chainComposite);
//         }

//         // Update lastTapTime for double-tap detection
//         lastTapTime = currentTime;

//         // For single touch, apply forces
//         if (touchCount === 1) {
//             const touch = touches[0];
//             const canvasRect = canvas.getBoundingClientRect();
//             const touchX = touch.clientX - canvasRect.left;
//             const centerX = canvasRect.width / 2;

//             const currentArm = currentControl === 'lowerArm' ? lowerArm : upperArm;

//             // Apply force based on touch position
//             if (touchX < centerX) {
//                 const message = 'Touch detected: Left side - Applying left force';
//                 updateOutputMessage(message); // Update index page
//                 Matter.Body.applyForce(currentArm, currentArm.position, { x: -0.05, y: 0 });
//             } else {
//                 const message = 'Touch detected: Right side - Applying right force';
//                 updateOutputMessage(message); // Update index page
//                 Matter.Body.applyForce(currentArm, currentArm.position, { x: 0.05, y: 0 });
//             }
//         }
//     }

//     function handleTouchEnd(event) {
//         event.preventDefault();
//         if (event.touches.length === 0) {
//             isSwitching = false; // Reset switching state if all fingers are lifted
//         }
//     }

//     function switchControl() {
//         if (currentControl === 'lowerArm') {
//             currentControl = 'upperArm';
//             // Update rendering
//             upperArm.render.fillStyle = '#ff6666'; // Highlight upper arm
//             upperArm.render.strokeStyle = '#1a1a1a';
//             lowerArm.render.fillStyle = 'transparent';
//             lowerArm.render.strokeStyle = '#1a1a1a';
//             updateOutputMessageForCurrentlySelectedLink('Currently controlling: Upper Arm');
//         } else {
//             currentControl = 'lowerArm';
//             // Update rendering
//             lowerArm.render.fillStyle = '#ff6666'; // Highlight lower arm
//             lowerArm.render.strokeStyle = '#1a1a1a';
//             upperArm.render.fillStyle = 'transparent';
//             upperArm.render.strokeStyle = '#1a1a1a';
//             updateOutputMessageForCurrentlySelectedLink('Currently controlling: Lower Arm');
//         }
//     }
// }

// Function to add touch controls
function addTouchControl(pendulum, canvas, engine, chainComposite) {
    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];
    let isSwitching = false; // Prevent rapid toggling
    let lastTapTime = 0; // To track time of the last tap
    let isDoubleTapped = false; // To track double-tap status

    // Add touchstart and touchmove event listeners to the canvas
    canvas.addEventListener('touchstart', handleTouch, { passive: false });
    canvas.addEventListener('touchmove', handleTouch, { passive: false });
    canvas.addEventListener('touchend', handleTouchEnd, { passive: false });

    // Watch the hidden input for changes
    const hiddenInput = document.getElementById('doubleTapValue');
    if (hiddenInput) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'value') {
                    isDoubleTapped = hiddenInput.value === 'true'; // Check if double-tap is true
                    if (isDoubleTapped) {
                        handleDoubleTap(); // Call the double-tap logic
                    }
                }
            });
        });

        observer.observe(hiddenInput, {
            attributes: true,
            attributeFilter: ['value'],
        });
    } else {
        console.error('Hidden input element not found');
    }

    function handleTouch(event) {
        event.preventDefault(); // Prevent default touch behavior (e.g., scrolling)
        const touches = event.touches;
        const touchCount = touches.length;

        // const currentTime = new Date().getTime();
        //  const timeSinceLastTap = currentTime - lastTapTime;

        // if (timeSinceLastTap < 300 && event.touches.length === 1) { // Detect double-tap
        //     isDoubleTapped = true; // Set the double-tap flag to true
        //     setTimeout(() => {
        //         isDoubleTapped = false; // Reset the flag after a short duration
        //     }, 500); // Keep the double-tap flag active for 500ms
        // }

        //  lastTapTime = currentTime; // Update last tap time
    }
    // Detect multi-touch


    function handleDoubleTap() {
      //  alert("double tapped");

        const countDownTimer = document.getElementById('countDownTimer');
        const countDown = document.getElementById('displayCountDown');
        const goalZone = document.getElementById('goalZone')

        if (countDown) {
            // Hide the `countDownTimer` initially
            goalZone.style.display = 'none';

            // Show the `countDownTimer` when the countdown starts
            countDownTimer.style.display = 'block';
            let countdownValue = 2; // Set the countdown start value (e.g., 3 seconds)

            // Update the countdown element to show the current value
            countDown.textContent = countdownValue;

            const countdownInterval = setInterval(() => {
                countdownValue--; // Decrease the countdown value
                countDown.textContent = countdownValue; // Update the display with the current value

                if (countdownValue <= 0) {
                    clearInterval(countdownInterval); // Stop the timer when it reaches 0
                    countDown.textContent = '0'; // Ensure it displays 0 explicitly
                }
            }, 1000); // 1-second interval
        }


        const randomZones = [];
        // Get the zone display elements
        const randomZone1Display = document.getElementById('randomZone1Display');
        const randomZone2Display = document.getElementById('randomZone2Display');
        const randomZone3Display = document.getElementById('randomZone3Display');
        const randomZone4Display = document.getElementById('randomZone4Display');

        // Add the values to the array if they exist and have text content
        if (randomZone1Display && randomZone1Display.textContent.trim()) {
            randomZones.push(randomZone1Display.textContent.trim());
        }
        if (randomZone2Display && randomZone2Display.textContent.trim()) {
            randomZones.push(randomZone2Display.textContent.trim());
        }
        if (randomZone3Display && randomZone3Display.textContent.trim()) {
            randomZones.push(randomZone3Display.textContent.trim());
        }
        if (randomZone4Display && randomZone4Display.textContent.trim()) {
            randomZones.push(randomZone4Display.textContent.trim());
        }

        // Show the array length in an alert box
        //   alert(`Number of Random Zones: ${randomZones.length}\nValues: ${randomZones.join(', ')}`);


        // Call removeChain if required (keeping your original functionality)
        removeChain(engine, chainComposite);

        // Add a 1.5-second delay using setTimeout
        setTimeout(() => {



            // Populate targetZones with the zones we need to match
            links.forEach((link, index) => {
                const zoneValueElement = document.getElementById(`armZone${index}Output`); // Corrected element ID
                if (zoneValueElement) {     //armZone1Output

                    const zoneValue = zoneValueElement.textContent.trim();

                    // Find and remove matching zone from randomZones
                    const matchingIndex = randomZones.indexOf(zoneValue);
                    if (matchingIndex !== -1) {
                        randomZones.splice(matchingIndex, 1); // Remove the matching zone
                        //  alert(`Match found and removed: ${zoneValue}`);
                    }
                }
            })

            if (randomZones == 0) {
                score += 1;
                showResultModal(true, 'Success! Pendulum(s) landed in their correct zones. Score +1', playerName, playerDifficultyLevel, numberOfLinks, score, length, width, airFriction);

            } else {
                let mismatchDetails = 'Mismatch!\nExpected Zones:\n';
                randomZones.forEach((zone, index) => {
                    mismatchDetails += `Zone ${index + 1}: ${zone}\n`;
                });


                // var mismatchMessage = `Mismatch!
                //  Upper Arm Zone: ${upperArmZone.innerText}, Expected: ${randomZone1}
                // Lower Arm Zone: ${lowerArmZone.innerText}, Expected: ${randomZone2}`;
                // //         alert(`);
                showResultModal(false, mismatchDetails, playerName, playerDifficultyLevel, numberOfLinks, score, length, width, airFriction);

            }

            // Progression logic: Increase level after every 5 successful scores
            if (score > 0 && score % 5 === 0) {
                updateLevel(level + 1);
            }
        }, 2000); // 1.5-second delay

        // Determine zones based on the current level
        determineZonesForLevel();

    }

    // // For single touch, apply forces
    // if (touchCount === 1) {
    //     const touch = touches[0];
    //     const canvasRect = canvas.getBoundingClientRect();
    //     const touchX = touch.clientX - canvasRect.left;
    //     const centerX = canvasRect.width / 2;

    //     const currentArm = currentControl === 'lowerArm' ? lowerArm : upperArm;

    //     // Apply force based on touch position
    //     if (touchX < centerX) {
    //         const message = 'Touch detected: Left side - Applying left force';
    //         //   console.log(message);
    //         updateOutputMessage(message); // Update index page
    //         Matter.Body.applyForce(currentArm, currentArm.position, { x: -0.05, y: 0 });
    //     } else {
    //         const message = 'Touch detected: Right side - Applying right force';
    //         //  console.log(message);
    //         updateOutputMessage(message); // Update index page
    //         Matter.Body.applyForce(currentArm, currentArm.position, { x: 0.05, y: 0 });
    //     }
    // }


    function handleTouchEnd(event) {
        event.preventDefault();
        if (event.touches.length === 0) {
            isSwitching = false; // Reset switching state if all fingers are lifted
        }
    }

    function switchControl() {
        if (currentControl === 'lowerArm') {
            currentControl = 'upperArm';
            //    console.log('Switched control to upper arm');
            // Update rendering
            upperArm.render.fillStyle = '#ff6666'; // Highlight upper arm
            upperArm.render.strokeStyle = '#1a1a1a';
            lowerArm.render.fillStyle = 'transparent';
            lowerArm.render.strokeStyle = '#1a1a1a';
            updateOutputMessageForCurrentlySelectedLink('Currently controlling: Upper Arm');
        } else {
            currentControl = 'lowerArm';
            //   console.log('Switched control to lower arm');
            // Update rendering
            lowerArm.render.fillStyle = '#ff6666'; // Highlight lower arm
            lowerArm.render.strokeStyle = '#1a1a1a';
            upperArm.render.fillStyle = 'transparent';
            upperArm.render.strokeStyle = '#1a1a1a';
            updateOutputMessageForCurrentlySelectedLink('Currently controlling: Lower Arm');
        }
    }
}

// Function to update output on the index page
function updateOutputMessage(message) {
    const keyPressOutElement = document.getElementById('keyPressOut');
    if (keyPressOutElement) {
        keyPressOutElement.textContent = `Event Message: ${message}`;
    }
}

// Function to update output on the index page
function updateOutputMessageForCurrentlySelectedLink(message) {
    const keyPressOutElement = document.getElementById('currentLink');
    if (keyPressOutElement) {
        keyPressOutElement.textContent = `Event Message: ${message}`;
    }
}

// Function for adding moving the pendulum by the ai
function addAIMovement(pendulum) {
    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];



    // Function to move the pendulum based on the AI's prediction
    // function movePendulum(prediction) {
    //     const currentArm = currentControl === 'lowerArm' ? lowerArm : upperArm;

    //     // Apply force based on the prediction
    //     if (prediction === 0) {
    //         const message = 'AI Prediction: Left - Applying left force';
    //         console.log(message);
    //         updateOutputMessage(message); // Update index page
    //         Matter.Body.applyForce(currentArm, currentArm.position, { x: -0.05, y: 0 });
    //     } else if (prediction === 1) {
    //         const message = 'AI Prediction: Right - Applying right force';
    //         console.log(message);
    //         updateOutputMessage(message); // Update index page
    //         Matter.Body.applyForce(currentArm, currentArm.position, { x: 0.05, y: 0 });
    //     }
    // }

    // // Listen for messages from the AI
    // window.addEventListener('message', (event) => {
    //     if (event.data.type === 'aiPrediction') {
    //         movePendulum(event.data.prediction);
    //     }
    // });
}

// Function to apply torque based on best action
function applyBestActionTorque(pendulum, bestAction) {
    const lowerArm = pendulum.bodies[1];
    const torqueMagnitude = 0.05; // Torque magnitude (N m)

    if (bestAction === 0) { // Apply -1 torque
        //  console.log("Best Action: Apply -1 torque");
        Matter.Body.applyForce(lowerArm, lowerArm.position, { x: -torqueMagnitude, y: 0 });
    } else if (bestAction === 1) { // Apply 0 torque
        //    console.log("Best Action: Apply 0 torque");
        // No force applied
    } else if (bestAction === 2) { // Apply 1 torque
        //     console.log("Best Action: Apply 1 torque");
        Matter.Body.applyForce(lowerArm, lowerArm.position, { x: torqueMagnitude, y: 0 });
    }
}


// Function to add keyboard control
function addKeyboardControl(pendulum, engine, chainComposite) {
    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];
    const pauseDuration = 100; // 100ms pause duration
    let isPaused = false;
    let isKeyPressed = false;

    document.addEventListener('keydown', (event) => {
        //    isPaused = true;
        //    isKeyPressed = true;
        // Start the while loop that sets pendulum state to null
        // Start the asynchronous loop in a separate function
        // handleKeyPress();
        //  console.log(`Key pressed: ${event.key}`);
        const key = event.key;
        const currentArm = currentControl === 'lowerArm' ? lowerArm : upperArm;

        if (key === 'ArrowLeft') {
            const message = 'Arrow key pressed: Left - Applying left force';
            //     console.log(message);
            updateOutputMessage(message); // Update index page
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: -0.05, y: 0 });
        } else if (key === 'ArrowRight') {
            const message = 'Arrow key pressed: Right - Applying right force';
            //   console.log(message);
            updateOutputMessage(message); // Update index page
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: 0.05, y: 0 });
        } else if (key === 'a') {
            currentControl = 'lowerArm'; // Switch to lower arm
            //  console.log('Switched control to lower arm');
            // Fill the selected arm with lighter red
            lowerArm.render.fillStyle = '#ff6666'; // Lighter red
            lowerArm.render.strokeStyle = '#1a1a1a';
            // Reset the other arm
            upperArm.render.fillStyle = 'transparent';
            upperArm.render.strokeStyle = '#1a1a1a';
            updateOutputMessageForCurrentlySelectedLink('Currently controlling: Lower Arm');

            // Update the zone display on the webpage
            // document.querySelector('#currentLink').textContent = 'Currently controlling: Lower Arm';
            // if (currentLink) {
            //     currentLink.innerText = `Currently controlling: Lower Arm`;
            // }
        } else if (key === 's') {
            currentControl = 'upperArm'; // Switch to upper arm
            //   console.log('Switched control to upper arm');
            // Fill the selected arm with lighter red
            upperArm.render.fillStyle = '#ff6666'; // Lighter red
            upperArm.render.strokeStyle = '#1a1a1a';
            // Reset the other arm
            lowerArm.render.fillStyle = 'transparent';
            lowerArm.render.strokeStyle = '#1a1a1a';
            // Update the zone display on the webpage
            updateOutputMessageForCurrentlySelectedLink('Currently controlling: Upper Arm');
            // document.querySelector('#currentLink').textContent = 'Currently controlling: Upper Arm';
            // if (currentLink) {
            //     currentLink.innerText = `Currently controlling: Upper Arm`;
            // }
        } else if (key === 'c') {
            const countDownTimer = document.getElementById('countDownTimer');
            const countDown = document.getElementById('displayCountDown');
            const goalZone = document.getElementById('goalZone')

            if (countDown) {
                // Hide the `countDownTimer` initially
                goalZone.style.display = 'none';

                // Show the `countDownTimer` when the countdown starts
                countDownTimer.style.display = 'block';
                let countdownValue = 2; // Set the countdown start value (e.g., 3 seconds)

                // Update the countdown element to show the current value
                countDown.textContent = countdownValue;

                const countdownInterval = setInterval(() => {
                    countdownValue--; // Decrease the countdown value
                    countDown.textContent = countdownValue; // Update the display with the current value

                    if (countdownValue <= 0) {
                        clearInterval(countdownInterval); // Stop the timer when it reaches 0
                        countDown.textContent = '0'; // Ensure it displays 0 explicitly
                    }
                }, 1000); // 1-second interval
            }


            const randomZones = [];
            // Get the zone display elements
            const randomZone1Display = document.getElementById('randomZone1Display');
            const randomZone2Display = document.getElementById('randomZone2Display');
            const randomZone3Display = document.getElementById('randomZone3Display');
            const randomZone4Display = document.getElementById('randomZone4Display');

            // Add the values to the array if they exist and have text content
            if (randomZone1Display && randomZone1Display.textContent.trim()) {
                randomZones.push(randomZone1Display.textContent.trim());
            }
            if (randomZone2Display && randomZone2Display.textContent.trim()) {
                randomZones.push(randomZone2Display.textContent.trim());
            }
            if (randomZone3Display && randomZone3Display.textContent.trim()) {
                randomZones.push(randomZone3Display.textContent.trim());
            }
            if (randomZone4Display && randomZone4Display.textContent.trim()) {
                randomZones.push(randomZone4Display.textContent.trim());
            }

            // Show the array length in an alert box
            //   alert(`Number of Random Zones: ${randomZones.length}\nValues: ${randomZones.join(', ')}`);


            // Call removeChain if required (keeping your original functionality)
            removeChain(engine, chainComposite);

            // Add a 1.5-second delay using setTimeout
            setTimeout(() => {



                // Populate targetZones with the zones we need to match
                links.forEach((link, index) => {
                    const zoneValueElement = document.getElementById(`armZone${index}Output`); // Corrected element ID
                    if (zoneValueElement) {     //armZone1Output

                        const zoneValue = zoneValueElement.textContent.trim();

                        // Find and remove matching zone from randomZones
                        const matchingIndex = randomZones.indexOf(zoneValue);
                        if (matchingIndex !== -1) {
                            randomZones.splice(matchingIndex, 1); // Remove the matching zone
                            //  alert(`Match found and removed: ${zoneValue}`);
                        }
                    }
                })

                if (randomZones == 0) {
                    score += 1;
                    showResultModal(true, 'Success! Pendulum(s) landed in their correct zones. Score +1', playerName, playerDifficultyLevel, numberOfLinks, score, length, width, airFriction);

                } else {
                    let mismatchDetails = 'Mismatch!\nExpected Zones:\n';
                    randomZones.forEach((zone, index) => {
                        mismatchDetails += `Zone ${index + 1}: ${zone}\n`;
                    });


                    // var mismatchMessage = `Mismatch!
                    //  Upper Arm Zone: ${upperArmZone.innerText}, Expected: ${randomZone1}
                    // Lower Arm Zone: ${lowerArmZone.innerText}, Expected: ${randomZone2}`;
                    // //         alert(`);
                    showResultModal(false, mismatchDetails, playerName, playerDifficultyLevel, numberOfLinks, score, length, width, airFriction);

                }

                // Progression logic: Increase level after every 5 successful scores
                if (score > 0 && score % 5 === 0) {
                    updateLevel(level + 1);
                }
            }, 2000); // 1.5-second delay

            // Determine zones based on the current level
            determineZonesForLevel();
        }
    });

    // document.addEventListener('keyup', (event) => {
    //    isKeyPressed = false;
    //    isPaused = false;
    //  const pendulumStateInput = document.getElementById('stateInput');
    //  pendulumStateInput.disabled = false;  // Disable the element
    //  pendulumStateInput.readOnly = false;  // Make it read-only as an extra measure
    // });

    // Separate async function for handling the loop
    //  async function handleKeyPress() {
    //     while (isKeyPressed) {
    //         const pendulumStateInput = document.getElementById('stateInput');
    //   pendulumStateInput.value = 'stop';
    //  pendulumStateInput.disabled = true;  // Disable the element
    //  pendulumStateInput.readOnly = true;  // Make it read-only as an extra measure
    //   await new Promise(resolve => setTimeout(resolve, pauseDuration)); // 100ms delay between updates
    //     }
    // }
}

// Draw Zone Indicators on the Grid
function drawZoneLabels(context, canvasWidth, canvasHeight) {
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;

    context.font = '16px Arial';
    context.fillStyle = '#555';

    // // Top-Left
    // context.fillText('Top-Left', centerX / 2 - 30, centerY / 2);

    // // Top-Right
    // context.fillText('Top-Right', centerX + centerX / 2 - 40, centerY / 2);

    // // Bottom-Left
    // context.fillText('Bottom-Left', centerX / 2 - 40, centerY + centerY / 2);

    // // Bottom-Right
    // context.fillText('Bottom-Right', centerX + centerX / 2 - 50, centerY + centerY / 2);

    // Quadrant Labels
    context.font = '14px Arial';

    // Q1: Top-Right (Blue)
    context.fillStyle = 'blue';
    // context.fillText('Top-Right Quadrant (0° to 90°)', canvasWidth * 0.75 - 50, canvasHeight * 0.25 - 10);
    context.fillText('Top-Right Quad', canvasWidth * 0.75 - 50, canvasHeight * 0.25 - 10);


    // Q2: Top-Left (Orange)
    context.fillStyle = 'orange';
    // context.fillText('Top-Left Quadrant (90° to 180°)', canvasWidth * 0.25 - 50, canvasHeight * 0.25 - 10);
    context.fillText('Top-Left Quad', canvasWidth * 0.25 - 50, canvasHeight * 0.25 - 10);

    // Q3: Bottom-Left (Green)
    context.fillStyle = 'green';
    // context.fillText('Bottom-Left Quadrant (180° to 270°)', canvasWidth * 0.25 - 50, canvasHeight * 0.75 - 10);
    context.fillText('Bottom-Left Quad', canvasWidth * 0.25 - 50, canvasHeight * 0.75 - 10);


    // Q4: Bottom-Right (Red)
    context.fillStyle = 'red';
    //context.fillText('Bottom-Right (270° to 360°)', canvasWidth * 0.75 - 50, canvasHeight * 0.75 - 10);
    context.fillText('Bottom-Right Quad', canvasWidth * 0.75 - 50, canvasHeight * 0.75 - 10);
}

function getZone(position, canvasWidth, canvasHeight) {
    const centerX = canvasWidth / 2; // X-axis center
    const centerY = canvasHeight / 2; // Y-axis center

    if (position.x < centerX && position.y < centerY) {
        return 'Top-Left Quadrant';
    } else if (position.x >= centerX && position.y < centerY) {
        return 'Top-Right Quadrant';
    } else if (position.x < centerX && position.y >= centerY) {
        return 'Bottom-Left Quadrant';
    } else if (position.x >= centerX && position.y >= centerY) {
        return 'Bottom-Right Quadrant';
    }
}

function detectCircleFromTrail(trail, tolerance = 5, minDistance = 50) {
    if (trail.length < 10) return false; // Not enough points to detect a circle

    const startPoint = trail[0].position; // Starting point of the trail

    // Check the last point in the trail
    const endPoint = trail[trail.length - 1].position;

    // Calculate distance between start and end points
    const distance = Math.sqrt(
        Math.pow(endPoint.x - startPoint.x, 2) +
        Math.pow(endPoint.y - startPoint.y, 2)
    );

    // Check if the trail has returned near the start point
    if (distance < tolerance) {
        // Verify the pendulum has traveled enough distance to form a circle
        let totalTravelDistance = 0;
        for (let i = 1; i < trail.length; i++) {
            const prev = trail[i - 1].position;
            const current = trail[i].position;

            totalTravelDistance += Math.sqrt(
                Math.pow(current.x - prev.x, 2) +
                Math.pow(current.y - prev.y, 2)
            );
        }

        // If the total travel distance is large enough, count as a circle
        if (totalTravelDistance > minDistance) {
            return true; // Circle detected
        }
    }

    return false; // No circle detected
}


// function addKeyboardControl(pendulum) {
//     const lowerArm = pendulum.bodies[1]; // Access the lower arm body
//     document.addEventListener('keydown', (event) => {
//         console.log(`Key pressed: ${event.key}`);
//         const key = event.key;
//         if (key === 'ArrowLeft') {
//             console.log('Applying left force');
//             Matter.Body.applyForce(lowerArm, lowerArm.position, { x: -0.05, y: 0 });
//         } else if (key === 'ArrowRight') {
//             console.log('Applying right force');
//             Matter.Body.applyForce(lowerArm, lowerArm.position, { x: 0.05, y: 0 });
//         }
//     });
// }

var Simulation = Simulation || {};

Simulation.doublePendulum = async (containerId, centerX, centerY, websitePlayerScore = 0, websitePlayerLevel = 1, websitePendulumWidth = 25, websitePendulumLength = 100, websitePlayerDifficultyLevel = 'easy', websitePlayerName = 'Guest', pendulumNumberValue = 2, airFrictionValue = 0) => {
    const { Engine, Events, Render, Runner, Body, Composite, Composites, Constraint, MouseConstraint, Mouse, Bodies, Vector } = Matter;

    // Initialize the game
    determineZonesForLevel(websitePlayerDifficultyLevel);
    // alert(`websitePlayerScore is ${websitePlayerScore}`)
    updateScore(websitePlayerScore === null ? 0 : websitePlayerScore);
    updateLevel(websitePlayerLevel === null ? 1 : websitePlayerLevel);
    length = websitePendulumLength;
    width = websitePendulumWidth;
    playerName = websitePlayerName;
    playerDifficultyLevel = websitePlayerDifficultyLevel;
    numberOfLinks = pendulumNumberValue;
    airFriction = airFrictionValue;
    // alert(`Debug Values:
    //     Player Score: ${websitePlayerScore === null ? 0 : websitePlayerScore}
    //     Player Level: ${websitePlayerLevel === null ? 1 : websitePlayerLevel}
    //     websitePlayerName: ${websitePlayerName} 
    //     Difficulty Level: ${websitePlayerDifficultyLevel}
    //     Number of Pendulum Links: ${pendulumNumberValue}
    //     Pendulum Length: ${websitePendulumLength}
    //     Pendulum Width: ${websitePendulumWidth}
    //      Airfriction Value: ${airFrictionValue}`)
    //     ;


    //  console.log("centerX = " + centerX);
    //  console.log("centerY = " + centerY);

    // Get the container element by ID
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container with ID "${containerId}" not found.`);
        return;
    }

    // Create engine
    const engine = Engine.create();
    const { world } = engine;

    // Create renderer
    const render = Render.create({
        element: container,
        engine: engine,
        options: {
            width: container.clientWidth,
            height: container.clientHeight,
            wireframes: false,
            background: '#fafafa'
        }
    });

    Render.run(render);

    // Function to draw a grid with x and y axes and an arrow at the bottom of the y-axis
    const drawGrid = (context, width, height, gridSize = 20) => {
        // Calculate center of the canvas
        const centerX = width / 2;
        const centerY = height / 2;

        context.strokeStyle = '#e0e0e0';
        context.lineWidth = 1;

        // Draw vertical grid lines
        for (let x = gridSize; x < width; x += gridSize) {
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
        }

        // Draw horizontal grid lines
        for (let y = gridSize; y < height; y += gridSize) {
            context.beginPath();
            context.moveTo(0, y);
            context.lineTo(width, y);
            context.stroke();
        }

        // Draw x-axis
        context.strokeStyle = '#333';
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(0, height / 2);
        context.lineTo(width, height / 2);
        context.stroke();

        // Draw y-axis
        context.beginPath();
        context.moveTo(width / 2, 0);
        context.lineTo(width / 2, height);
        context.stroke();

        // Add degree markings
        context.font = '10px Arial';
        context.fillStyle = '#000';

        for (let angle = 0; angle < 360; angle += 10) {
            const radians = (angle * Math.PI) / 180;

            // Calculate position for the degree marker
            const markerX = centerX + Math.cos(radians) * (gridSize * 8); // Adjust radius as needed
            const markerY = centerY - Math.sin(radians) * (gridSize * 8); // Adjust radius as needed

            // Position for text slightly outward
            const textX = centerX + Math.cos(radians) * (gridSize * 9);
            const textY = centerY - Math.sin(radians) * (gridSize * 9);

            // Draw small line for degree marker
            context.beginPath();
            context.moveTo(centerX + Math.cos(radians) * (gridSize * 7), centerY - Math.sin(radians) * (gridSize * 7));
            context.lineTo(markerX, markerY);
            context.stroke();

            // Draw degree text
            context.fillText(angle.toString(), textX - 5, textY + 5);
        }





        // Draw an arrow at the bottom of the y-axis
        context.fillStyle = '#333';
        context.beginPath();
        context.moveTo(width / 2, height);           // Bottom center of y-axis
        context.lineTo(width / 2 - 5, height - 10);  // Left side of arrowhead
        context.lineTo(width / 2 + 5, height - 10);  // Right side of arrowhead
        context.closePath();
        context.fill();

        // Optional axis labels
        context.font = '14px Arial';
        context.fillStyle = '#333';
        context.fillText('Y', width / 2 + 5, 15);    // Label Y near the top
        context.fillText('X', width - 15, height / 2 - 5);  // Label X near the right

        // // Draw quadrant labels with degree ranges
        // context.font = '12px Arial';
        // context.fillStyle = 'blue';
        // context.fillText('Q1 (0° to 90°)', width * 0.75, height * 0.25);  // Quadrant 1
        // context.fillStyle = 'orange';
        // context.fillText('Q2 (90° to 180°)', width * 0.25, height * 0.25);  // Quadrant 2
        // context.fillStyle = 'green';
        // context.fillText('Q3 (180° to 270°)', width * 0.25, height * 0.75);  // Quadrant 3
        // context.fillStyle = 'red';
        // context.fillText('Q4 (270° to 360°)', width * 0.75, height * 0.75);  // Quadrant 4

        // Color the quadrants with soft, distinct colors
        context.fillStyle = 'rgba(255, 182, 193, 0.2)';    // Soft pink
        context.fillRect(width / 2, 0, width / 2, height / 2);  // Q1

        context.fillStyle = 'rgba(173, 216, 230, 0.2)';    // Soft blue
        context.fillRect(0, 0, width / 2, height / 2);      // Q2

        context.fillStyle = 'rgba(144, 238, 144, 0.2)';    // Soft green
        context.fillRect(0, height / 2, width / 2, height / 2);  // Q3

        context.fillStyle = 'rgba(255, 218, 185, 0.2)';    // Soft peach
        context.fillRect(width / 2, height / 2, width / 2, height / 2);  // Q4


    };

    // Create runner
    const runner = Runner.create();
    Runner.run(runner, engine);

    // Add bodies
    const group = Body.nextGroup(true);
    // we change this to change the number of links

    // const length = 100;    // decreased from 200 to 100
    // const width = 25;

    // const group = Body.nextGroup(true);
    // const length = 150; // Change from 200 to whatever length you want
    // const width = 25;   // This is the width of each link

    //const length = 100;
    //const width = 25;



    //const length = 100;
    //const width = 25;
    // const numberOfLinks = 3;
    // const pendulum = Composites.stack(300, 160, 2, 1, -20, 0, (x, y) =>

    const pendulum = Composites.stack(300, 160, numberOfLinks, 1, -20, 0, (x, y) =>
        Bodies.rectangle(x, y, length, width, {
            collisionFilter: { group },
            frictionAir: airFriction,
            //frictionAir: 0,
            chamfer: 5,
            render: {
                fillStyle: 'transparent',
                lineWidth: 1
            }
        })
    );

    engine.gravity.scale = 0.002;

    // Store the chain composite when you create it
    const chainComposite = Composites.chain(pendulum, 0.45, 0, -0.45, 0, {
        stiffness: 0.9,
        length: 50,
        angularStiffness: 0.7,
        render: {
            strokeStyle: '#4a485b'
        }
    });

    // Composites.chain(pendulum, 0.45, 0, -0.45, 0, {
    //     stiffness: 0.9,
    //     length: 50,
    //     angularStiffness: 0.7,
    //     render: {
    //         strokeStyle: '#4a485b'
    //     }
    // });

    // console.log("centerX = " + centerX);
    // console.log("centerY = " + centerY);


    // Calculate grid center coordinates
    // const gridCenterX = 1296 / 2;  // = 648
    // const gridCenterY = 904 / 2;   // = 452

    const gridCenterX = centerX;  // = 648
    const gridCenterY = centerY;   // = 452

    Composite.add(pendulum, Constraint.create({
        bodyB: pendulum.bodies[0],
        pointB: { x: -length * 0.42, y: 0 },
        // pointB: { x: 19, y: 0 },  // Fixed smaller value
        pointA: { x: gridCenterX, y: gridCenterY },
        stiffness: 0.9,
        length: 0,
        render: {
            strokeStyle: '#4a485b'
        }
    }));


    for (let i = 0; i < numberOfLinks; i++) {
        links.push(pendulum.bodies[i]);
    }

    // Show all the links in alert boxes
    // links.forEach((link, index) => {
    //     alert(`Link ${index + 1}:\nPosition: (${link.position.x.toFixed(2)}, ${link.position.y.toFixed(2)})\nAngle: ${link.angle.toFixed(2)} radians`);
    // });

    // so this where we add arm to control, lowerarm is the control now
    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];

    // Rotate the lower arm 90 degrees (Math.PI/2) around its position
    Body.rotate(lowerArm, Math.PI / 2, {
        x: lowerArm.position.x,
        y: lowerArm.position.y
    });

    // Higher Values = more damping, for faster equilibrium
    // lowerArm.friction = 0.8;      // Increased from 0.3 to 0.8 for more surface friction
    // lowerArm.frictionAir = 0.3;   // Increased from 0.1 to 0.3 for more air resistance
    // lowerArm.restitution = 0.05;  // Decreased from 0.1 to 0.05 for less bouncing

    // Set the pendulum to a stable, vertical position with no initial movement
    // Body.setPosition(lowerArm, { x: 300, y: 160 + length });
    // Body.setAngle(lowerArm, 0);
    // Body.setVelocity(lowerArm, { x: 0, y: 0 });
    // Body.setAngularVelocity(lowerArm, 0);

    Composite.add(world, pendulum);

    const trail = [];
    let lowerArmCircleCount = 0; // Count of lower arm circles
    let upperArmCircleCount = 0; // Count of upper arm circles

    Events.on(render, 'afterRender', () => {
        //    const observation = getObservationFromPendulum(lowerArm, upperArm);
        //     document.getElementById('hdnPendulumState').value = JSON.stringify(observation);

        // document.getElementById('hdnPendulumState').value = JSON.stringify({
        //     state: 'example',
        //     data: 123
        // });
        //  console.log("Current Observation:", observation);
        // set the hidden value 

        // Create or get the hidden input element
        // let hiddenElement = document.getElementById('hiddenValue');
        // hiddenElement.value = observation;
        // console.log("hiddenElement value is ", hiddenElement.value);

        // Send the observation data to the WebSocket server
        // logPendulumState(lowerArm, upperArm);

        // Check alignment
        const alignmentInfo = checkPendulumsAlignment(upperArm, lowerArm);
        // updateAlignmentDisplay(alignmentInfo);
        // if (alignmentInfo.isAligned) {
        //     //      console.log('Pendulums are aligned!');
        //     //     console.log('Upper arm angle:', alignmentInfo.upperDegrees.toFixed(2));
        //     //      console.log('Lower arm angle:', alignmentInfo.lowerDegrees.toFixed(2));

        //     // Optional: Update UI to show alignment
        //     const alignmentDisplay = document.getElementById('alignmentDisplay');
        //     alignmentDisplay.innerHTML = `
        //     <div style="color: green;">
        //         Pendulums Aligned!
        //         <br>
        //         Upper: ${alignmentInfo.upperDegrees.toFixed(2)}°
        //         <br>
        //         Lower: ${alignmentInfo.lowerDegrees.toFixed(2)}°
        //     </div>
        // `;

        if (alignmentInfo.isAligned) {
            //      console.log('Pendulums are aligned!');
            //     console.log('Upper arm angle:', alignmentInfo.upperDegrees.toFixed(2));
            //      console.log('Lower arm angle:', alignmentInfo.lowerDegrees.toFixed(2));

            // Optional: Update UI to show alignment
            const alignmentDisplay = document.getElementById('alignmentDisplay');
            alignmentDisplay.innerHTML = `
            <div style="color: green;">
                Pend Aligned!
            </div>
        `;

        } else {
            //   console.log('Pendulums are not aligned.');
            // Optional: Update UI to show non-alignment
            const alignmentDisplay = document.getElementById('alignmentDisplay');
            alignmentDisplay.innerHTML = `
            <div style="color: red;">
                Not Aligned!
            </div>
        `;
        }

        // Draw the grid
        drawGrid(render.context, render.options.width, render.options.height);

        // Update trails for both arms
        const lowerArmTrail = trails.lowerArm;
        const upperArmTrail = trails.upperArm;




        // If you want to see the length of each trail
        // console.log('Lower Arm Trail Length:', lowerArmTrail.length);
        // console.log('Upper Arm Trail Length:', upperArmTrail.length);

        // Store the position of the current controlled link
        if (currentControl === 'lowerArm') {
            lowerArmTrail.unshift({
                position: Vector.clone(lowerArm.position),
                speed: lowerArm.speed,
            });
        } else if (currentControl === 'upperArm') {
            upperArmTrail.unshift({
                position: Vector.clone(upperArm.position),
                speed: upperArm.speed,
            });
        }

        // // Print lower arm trail
        // console.log('Lower Arm Trail:', lowerArmTrail.map(point => ({
        //     x: point.position.x.toFixed(2),
        //     y: point.position.y.toFixed(2),
        //     speed: point.speed.toFixed(2)
        // })));

        // // Print upper arm trail
        // console.log('Upper Arm Trail:', upperArmTrail.map(point => ({
        //     x: point.position.x.toFixed(2),
        //     y: point.position.y.toFixed(2),
        //     speed: point.speed.toFixed(2)
        // })));
        // Check for circles in the lower arm trail
        if (detectCircleFromTrail(trails.lowerArm)) {
            //  console.log('Lower arm completed a circle!');
            lowerArmCircleCount++;
            trails.lowerArm.length = 0; // Reset trail after detecting a circle
            // Update the lower arm circle count on the page
            const lowerArmCircleElement = document.getElementById('lowerArmCircleCount');
            if (lowerArmCircleElement) {
                lowerArmCircleElement.textContent = `Lower Arm Circles: ${lowerArmCircleCount}`;
            }
        }

        // Check for circles in the upper arm trail
        if (detectCircleFromTrail(trails.upperArm)) {
            //   console.log('Upper arm completed a circle!');
            upperArmCircleCount++;
            trails.upperArm.length = 0; // Reset trail after detecting a circle
            // Update the upper arm circle count on the page
            const upperArmCircleElement = document.getElementById('upperArmCircleCount');
            if (upperArmCircleElement) {
                upperArmCircleElement.textContent = `Over Bar Count: ${upperArmCircleCount}`;
            }
        }

        // Limit trail length
        // if (lowerArmTrail.length > 800) lowerArmTrail.pop();
        // if (upperArmTrail.length > 2000) upperArmTrail.pop();

        // Draw the pendulum trail
        // trail.unshift({
        //     position: Vector.clone(upperArm.position),
        //     speed: upperArm.speed
        // });

        Render.startViewTransform(render);
        render.context.globalAlpha = 0.7;

        // trail.forEach((pointInfo) => {
        //     const { position: point, speed } = pointInfo;
        //     const hue = 250 + Math.round((1 - Math.min(1, speed / 10)) * 170);
        //     render.context.fillStyle = `hsl(${hue}, 100%, 55%)`;
        //     render.context.fillRect(point.x, point.y, 2, 2);
        // });

        // Draw lowerArm trail
        lowerArmTrail.forEach((pointInfo) => {
            const { position: point, speed } = pointInfo;
            const hue = 250 + Math.round((1 - Math.min(1, speed / 10)) * 170);
            render.context.fillStyle = `hsl(${hue}, 100%, 55%)`;
            render.context.fillRect(point.x, point.y, 2, 2);
        });

        // Draw upperArm trail
        upperArmTrail.forEach((pointInfo) => {
            const { position: point, speed } = pointInfo;
            const hue = 150 + Math.round((1 - Math.min(1, speed / 10)) * 170);
            render.context.fillStyle = `hsl(${hue}, 100%, 55%)`;
            render.context.fillRect(point.x, point.y, 2, 2);
        });

        render.context.globalAlpha = 1;
        Render.endViewTransform(render);

        // if (trail.length > 2000) {
        //     trail.pop();
        // }

        // Output the y-value of the pendulum's endpoint
        //console.log("Current Y position of lower arm endpoint:", lowerArm.position.y);

        // Calculate the y position of the pendulum's endpoint
        // const yEndpoint = calculateYEndpoint(lowerArm, pendulum.bodies[0], length, length);
        const yEndpoint = calculateYEndpoint(lowerArm, upperArm, length, length);
        // Update the Y position in the HTML element
        const yPositionElement = document.getElementById("yPositionOutput");
        if (yPositionElement) {
            yPositionElement.textContent =
                "Current Y position of lower arm endpoint: " + yEndpoint.toFixed(2);
        }
        drawZoneLabels(render.context, render.options.width, render.options.height);

        const lowerArmPos = lowerArm.position; // Access the pendulum's lower arm position
        const upperArmPos = upperArm.position;
        const zone = getZone(lowerArmPos, render.options.width, render.options.height); // Call getZone
        const lowerArmZone = getZone(lowerArmPos, render.options.width, render.options.height); // Call getZone
        const upperArmZone = getZone(upperArmPos, render.options.width, render.options.height); // Call getZone

        // Loop through links and get zone for each
        links.forEach((link, index) => {
            const zone = getZone(link.position, render.options.width, render.options.height);

            // Get the zone value element using the naming convention from createArmZones
            const zoneValueElement = document.getElementById(`armZone${index}Output`);


            // Update the zone value if the element exists
            if (zoneValueElement) {
                zoneValueElement.textContent = `${zone}`;
            }
            // console.log(`Link position: (${link.position.x}, ${link.position.y}), Zone: ${zone}`);
        });

        // Update the zone display on the webpage
        // const zoneOutput = document.getElementById('zoneOutput');
        // if (zoneOutput) {
        //     zoneOutput.textContent = `Current Zone: ${zone}`;
        // }

        // Update the zone display on the webpage
        const lowerArmZoneOutput = document.getElementById('lowerArmZoneOutput');
        if (lowerArmZoneOutput) {
            lowerArmZoneOutput.textContent = `${lowerArmZone}`;
        }

        // Update the zone display on the webpage
        const upperArmZoneOutput = document.getElementById('upperArmZoneOutput');
        if (upperArmZoneOutput) {
            upperArmZoneOutput.textContent = `${upperArmZone}`;
        }


        trackAlignment(upperArm, lowerArm);
    });

    // add mouse control
    const mouse = Mouse.create(render.canvas);
    const mouseConstraint = MouseConstraint.create(engine, {
        mouse,
        constraint: {
            stiffness: 0.2,
            render: {
                visible: false
            }
        }
    });

    Composite.add(world, mouseConstraint);

    // keep the mouse in sync with rendering
    render.mouse = mouse;

    // fit the render viewport to the scene
    Render.lookAt(render, {
        min: { x: 0, y: 0 },
        max: { x: container.clientWidth, y: container.clientHeight }
    });

    // Call the function to add keyboard control
    addKeyboardControl(pendulum, engine, chainComposite);
    addTouchControl(pendulum, render.canvas, engine, chainComposite);

    // Stop the runner to implement manual stepping
    Runner.stop(runner);

    const stepInterval = 1000 / 60; // 60 FPS step duration
    // const pauseDuration = 5000; // Pause for 5 seconds
    const pauseDuration = 50; // Pause for 5 seconds

    currentControl = 'upperArm'; // Switch to upper arm
    //   console.log('Switched control to upper arm');
    // Fill the selected arm with lighter red
    upperArm.render.fillStyle = '#ff6666'; // Lighter red
    upperArm.render.strokeStyle = '#1a1a1a';
    // Reset the other arm
    lowerArm.render.fillStyle = 'transparent';
    lowerArm.render.strokeStyle = '#1a1a1a';
    // Update the zone display on the webpage
    updateOutputMessageForCurrentlySelectedLink('Currently controlling: Upper Arm');


    // Wait 3 seconds to auto-toggle the AI switch
    //setTimeout(autoToggleAI, 3000);

    while (true) {
        Engine.update(engine, stepInterval);
        //logPendulumState(lowerArm, upperArm);
        Render.world(render);
        // const observation = getObservationFromPendulum(lowerArm, upperArm);
        // console.log("Current Observation:", observation);

        // Create or get the hidden input element
        // let hiddenElement = document.getElementById('hdnCurrentState');
        //  hiddenElement.value = observation;
        // console.log("hiddenElement value is ", hiddenElement.value);

        const observation = getObservationFromPendulum(lowerArm, upperArm);
        document.getElementById('hdnPendulumState').value = JSON.stringify(observation);
        // Step 3: Fetch best action from a model (e.g., AI prediction)
        const hiddenInputValue = document.getElementById('savedMessagesHidden').value; // this comes back from the server
        const parsedValue = JSON.parse(hiddenInputValue);
        //  console.log("Parsed Value for function:", parsedValue);

        const bestAction = getBestActionFromModel(parsedValue); // Replace with your AI logic
        // console.log("Best Action:", bestAction);

        // Step 4: Apply torque based on the best action
        // Apply torque based on the best action
        if (bestAction !== null) {
            applyBestActionTorque({ bodies: [upperArm, lowerArm] }, bestAction);
        }

        // Step 5: Pause simulation
        await pauseSimulation(pauseDuration);
        //   await pauseSimulation(pauseDuration);
    }
    

    return {
        engine,
        runner,
        render,
        canvas: render.canvas,
        stop: () => {
            Matter.Render.stop(render);
            Matter.Runner.stop(runner);
        }
    };
};


