const trails = {
    lowerArm: [],
    upperArm: [],
};

let currentControl = 'lowerArm'; // Track which link is currently controlled

// Add a function to handle pausing the simulation for a specific duration
function pauseSimulation(duration) {
    return new Promise((resolve) => setTimeout(resolve, duration));
}

function getBestActionFromModel(parsedValue) {
    // Safely navigate to best_action
    if (parsedValue && parsedValue.length > 0 && parsedValue[0]?.data?.best_action !== undefined) {
        return parsedValue[0].data.best_action;
    }
    // console.error("Invalid input structure: Unable to retrieve best_action.");
    return null; // Return null if best_action is not found
}

function detectChainConstraints(pendulum, engine) {
    const chainConstraints = engine.world.constraints.filter(constraint => {
        const bodyAInPendulum = pendulum.bodies.includes(constraint.bodyA);
        const bodyBInPendulum = pendulum.bodies.includes(constraint.bodyB);
        return bodyAInPendulum && bodyBInPendulum;
    });

    console.log('Chain Constraints:', chainConstraints);
    return chainConstraints;
}

function removeLowerArmFromChain(lowerArm, pendulum, engine) {
    const constraintsToRemove = engine.world.constraints.filter(constraint =>
        (constraint.bodyA === lowerArm || constraint.bodyB === lowerArm) &&
        pendulum.bodies.includes(constraint.bodyA) &&
        pendulum.bodies.includes(constraint.bodyB)
    );

    constraintsToRemove.forEach(constraint => {
        Matter.World.remove(engine.world, constraint);
        console.log('Removed Constraint:', constraint);
    });

    // Remove the lower arm from the composite
    Matter.Composite.remove(pendulum, lowerArm);

    // Add the lower arm back to the world if needed
    Matter.World.add(engine.world, lowerArm);

    console.log('Lower arm detached from chain.');
}



document.addEventListener('DOMContentLoaded', function () {
    console.log("in event listener");
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
function addTouchControl(pendulum, canvas) {
    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];
    let isSwitching = false; // Prevent rapid toggling

    // Add touchstart and touchmove event listeners to the canvas
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', handleTouchEnd);

    function handleTouch(event) {
        event.preventDefault(); // Prevent default touch behavior (e.g., scrolling)

        // Detect multi-touch
        const touches = event.touches;
        const touchCount = touches.length;

        if (touchCount === 2 && !isSwitching) {
            isSwitching = true; // Set switching state
            switchControl();
            setTimeout(() => (isSwitching = false), 500); // Prevent immediate toggling
            return;
        }

        // For single touch, apply forces
        if (touchCount === 1) {
            const touch = touches[0];
            const canvasRect = canvas.getBoundingClientRect();
            const touchX = touch.clientX - canvasRect.left;
            const centerX = canvasRect.width / 2;

            const currentArm = currentControl === 'lowerArm' ? lowerArm : upperArm;

            // Apply force based on touch position
            if (touchX < centerX) {
                const message = 'Touch detected: Left side - Applying left force';
                console.log(message);
                updateOutputMessage(message); // Update index page
                Matter.Body.applyForce(currentArm, currentArm.position, { x: -0.05, y: 0 });
            } else {
                const message = 'Touch detected: Right side - Applying right force';
                console.log(message);
                updateOutputMessage(message); // Update index page
                Matter.Body.applyForce(currentArm, currentArm.position, { x: 0.05, y: 0 });
            }
        }
    }

    function handleTouchEnd(event) {
        event.preventDefault();
        if (event.touches.length === 0) {
            isSwitching = false; // Reset switching state if all fingers are lifted
        }
    }

    function switchControl() {
        if (currentControl === 'lowerArm') {
            currentControl = 'upperArm';
            console.log('Switched control to upper arm');
            // Update rendering
            upperArm.render.fillStyle = '#ff6666'; // Highlight upper arm
            upperArm.render.strokeStyle = '#1a1a1a';
            lowerArm.render.fillStyle = 'transparent';
            lowerArm.render.strokeStyle = '#1a1a1a';
            updateOutputMessageForCurrentlySelectedLink('Currently controlling: Upper Arm');
        } else {
            currentControl = 'lowerArm';
            console.log('Switched control to lower arm');
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
        console.log("Best Action: Apply -1 torque");
        Matter.Body.applyForce(lowerArm, lowerArm.position, { x: -torqueMagnitude, y: 0 });
    } else if (bestAction === 1) { // Apply 0 torque
        console.log("Best Action: Apply 0 torque");
        // No force applied
    } else if (bestAction === 2) { // Apply 1 torque
        console.log("Best Action: Apply 1 torque");
        Matter.Body.applyForce(lowerArm, lowerArm.position, { x: torqueMagnitude, y: 0 });
    }
}

function detachLowerArm(pendulum, lowerArm, engine, chainConstraintss) {
    console.log('Detaching lower arm...');
    console.log("chainConstraintss are ", chainConstraintss);

   // const chainConstraints = engine.world.constraints;

    // // Filter and remove constraints involving the lower arm
    // const constraintsToRemove = chainConstraints.filter(
    //     (constraint) => constraint.bodyA === lowerArm || constraint.bodyB === lowerArm
    // );

    // constraintsToRemove.forEach((constraint) => {
    //     Matter.World.remove(engine.world, constraint);
    //     console.log('Removed Constraint:', constraint);
    // });

  //c  Matter.World.remove(engine.world, chainConstraintss[0]);

    // Remove the lower arm from the composite
  //  Matter.Composite.remove(pendulum, lowerArm);
 //   console.log('Lower arm removed from pendulum.');

    // Add the lower arm back to the world as an independent body
  //  Matter.World.add(engine.world, lowerArm);

    // Apply a force to make the lower arm move
    // const flyForce = { x: 0.05, y: -0.1 }; // Adjust the force as needed
    // Matter.Body.applyForce(lowerArm, lowerArm.position, flyForce);

    console.log('Lower arm detached and force applied.');
}


// Function to add keyboard control
function addKeyboardControl(pendulum, engine, upperToLowerConstraint, chainComposite) {
    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];
    // Iterate through constraints to detect those involving the pendulum's arms
const chainConstraints = engine.world.constraints.filter(constraint =>
    (constraint.bodyA === upperArm && constraint.bodyB === lowerArm) ||
    (constraint.bodyA === lowerArm && constraint.bodyB === upperArm)
);

console.log('Detected Chain Constraints:', chainConstraints);

    document.addEventListener('keydown', (event) => {
        console.log(`Key pressed: ${event.key}`);
        const key = event.key;
        const currentArm = currentControl === 'lowerArm' ? lowerArm : upperArm;

        if (key === 'ArrowLeft') {
            const message = 'Arrow key pressed: Left - Applying left force';
            console.log(message);
            updateOutputMessage(message); // Update index page
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: -0.05, y: 0 });
        } else if (key === 'ArrowRight') {
            const message = 'Arrow key pressed: Right - Applying right force';
            console.log(message);
            updateOutputMessage(message); // Update index page
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: 0.05, y: 0 });
        } else if (key === 'a') {
            currentControl = 'lowerArm'; // Switch to lower arm
            console.log('Switched control to lower arm');
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
            console.log('Switched control to upper arm');
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
            removeChain(engine, chainComposite);
        } else if (key === 'd') {
            console.log("Key 'd' pressed: Removing upper arm and spinning lower arm");

            console.log("upperToLowerContstraint is ", upperToLowerConstraint);

            // Remove the constraint between upperArm and lowerArm
            if (upperToLowerConstraint) {

                Matter.World.remove(engine.world, upperToLowerConstraint);
                console.log('Constraint between upper and lower arm removed.');
            }
            Matter.Composite.remove(pendulum, upperArm);

            // Remove the upper arm from the simulation
            Matter.World.remove(engine.world, upperArm);
           // upperArm.render.visible = false; // Optionally hide the upper arm visually
           Matter.Engine.update(engine);
            // Apply torque to the lower arm to make it spin
            const spinTorque = 0.1; // Adjust torque value for desired spin
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: spinTorque, y: 0 });

            updateOutputMessage('Upper arm removed, lower arm spinning!');

            detachLowerArm(lowerArm, pendulum, engine, chainConstraints);
            Matter.Body.setVelocity(lowerArm, { x: 1, y: -2 }); // Adjust the velocity
Matter.Body.setAngularVelocity(lowerArm, 0.2);     // Set some spin
           // detectChainConstraints(pendulum, engine);
// Call this function to find all constraints forming the chain
// const chainConstraints = detectChainConstraints(pendulum, engine);
// console.log('Detected Chain Constraints:', chainConstraints);

    // console.log('Detected Chain Constraints:', chainConstraints);

    // // Visualize the chain
    // chainConstraints.forEach(constraint => {
    //     constraint.render.strokeStyle = 'red';
    //     constraint.render.lineWidth = 3;
    // });

    // // Example: Remove the chain
    // chainConstraints.forEach(constraint => {
    //     Matter.World.remove(engine.world, constraint);
    // });

    // console.log('Remaining Constraints:', engine.world.constraints);
        }
    });

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
    context.fillText('Q1 (0° to 90°)', canvasWidth * 0.75 - 50, canvasHeight * 0.25 - 10);

    // Q2: Top-Left (Orange)
    context.fillStyle = 'orange';
    context.fillText('Q2 (90° to 180°)', canvasWidth * 0.25 - 50, canvasHeight * 0.25 - 10);

    // Q3: Bottom-Left (Green)
    context.fillStyle = 'green';
    context.fillText('Q3 (180° to 270°)', canvasWidth * 0.25 - 50, canvasHeight * 0.75 - 10);

    // Q4: Bottom-Right (Red)
    context.fillStyle = 'red';
    context.fillText('Q4 (270° to 360°)', canvasWidth * 0.75 - 50, canvasHeight * 0.75 - 10);
}

function getZone(position, canvasWidth, canvasHeight) {
    const centerX = canvasWidth / 2; // X-axis center
    const centerY = canvasHeight / 2; // Y-axis center

    if (position.x < centerX && position.y < centerY) {
        return 'Top-Left Zone';
    } else if (position.x >= centerX && position.y < centerY) {
        return 'Top-Right Zone';
    } else if (position.x < centerX && position.y >= centerY) {
        return 'Bottom-Left Zone';
    } else if (position.x >= centerX && position.y >= centerY) {
        return 'Bottom-Right Zone';
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

// Function to remove the chain
function removeChain(engine, chainComposite) {
    // Remove all constraints in the chain composite
    chainComposite.constraints.forEach(constraint => {
        Matter.World.remove(engine.world, constraint);
    });
    
    // Clear the constraints array
    chainComposite.constraints = [];
}

var Simulation = Simulation || {};

Simulation.doublePendulum = async (containerId, centerX, centerY) => {
    const { Engine, Events, Render, Runner, Body, Composite, Composites, Constraint, MouseConstraint, Mouse, Bodies, Vector } = Matter;

    console.log("centerX = " + centerX);
    console.log("centerY = " + centerY);

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

    const length = 100;
    const width = 25;

    const pendulum = Composites.stack(300, 160, 2, 1, -20, 0, (x, y) =>
        Bodies.rectangle(x, y, length, width, {
            collisionFilter: { group },
            frictionAir: 0,
            chamfer: 5,
            render: {
                fillStyle: 'transparent',
                lineWidth: 1
            }
        })
    );

    engine.gravity.scale = 0.002;

    // Create the chain
    // Composites.chain(pendulum, 0.45, 0, -0.45, 0, {
    //     stiffness: 0.9,
    //     length: 50,
    //     angularStiffness: 0.7,
    //     render: {
    //         strokeStyle: '#4a485b'
    //     }
    // });



// Store the chain composite when you create it
const chainComposite = Composites.chain(pendulum, 0.45, 0, -0.45, 0, {
    stiffness: 0.9,
    length: 50,
    angularStiffness: 0.7,
    render: {
        strokeStyle: '#4a485b'
    }
});

// Function to remove the chain
// function removeChain(engine, chainComposite) {
//     // Remove all constraints in the chain composite
//     chainComposite.constraints.forEach(constraint => {
//         Matter.World.remove(engine.world, constraint);
//     });
    
//     // Clear the constraints array
//     chainComposite.constraints = [];
// }




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
        pointA: { x: gridCenterX, y: gridCenterY },
        stiffness: 0.9,
        length: 0,
        render: {
            strokeStyle: '#4a485b'
        }
    }));

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
            const observation = getObservationFromPendulum(lowerArm, upperArm);
        //     document.getElementById('hdnPendulumState').value = JSON.stringify(observation);

        // document.getElementById('hdnPendulumState').value = JSON.stringify({
        //     state: 'example',
        //     data: 123
        // });
          console.log("Current Observation:", observation);
        // set the hidden value 

        // Create or get the hidden input element
        // let hiddenElement = document.getElementById('hiddenValue');
        // hiddenElement.value = observation;
        // console.log("hiddenElement value is ", hiddenElement.value);

        // Send the observation data to the WebSocket server
        // logPendulumState(lowerArm, upperArm);

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
            console.log('Lower arm completed a circle!');
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
            console.log('Upper arm completed a circle!');
            upperArmCircleCount++;
            trails.upperArm.length = 0; // Reset trail after detecting a circle
            // Update the upper arm circle count on the page
            const upperArmCircleElement = document.getElementById('upperArmCircleCount');
            if (upperArmCircleElement) {
                upperArmCircleElement.textContent = `Upper Arm Circles: ${upperArmCircleCount}`;
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
        const zone = getZone(lowerArmPos, render.options.width, render.options.height); // Call getZone

        // Update the zone display on the webpage
        const zoneOutput = document.getElementById('zoneOutput');
        if (zoneOutput) {
            zoneOutput.textContent = `Current Zone: ${zone}`;
        }
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

    const upperToLowerConstraint = Constraint.create({
        bodyA: upperArm,
        pointA: { x: 0, y: length / 2 }, // Attach to the bottom of the upper arm
        bodyB: lowerArm,
        pointB: { x: 0, y: -length / 2 }, // Attach to the top of the lower arm
        stiffness: 0.9,
        length: 0,
        render: {
            strokeStyle: '#4a485b'
        }
    });
    Composite.add(world, upperToLowerConstraint);

    // keep the mouse in sync with rendering
    render.mouse = mouse;

    // fit the render viewport to the scene
    Render.lookAt(render, {
        min: { x: 0, y: 0 },
        max: { x: container.clientWidth, y: container.clientHeight }
    });

    // Call the function to add keyboard control
    addKeyboardControl(pendulum, engine, upperToLowerConstraint, chainComposite);
    addTouchControl(pendulum, render.canvas);

    // Stop the runner to implement manual stepping
    Runner.stop(runner);

    const stepInterval = 1000 / 60; // 60 FPS step duration
    // const pauseDuration = 5000; // Pause for 5 seconds
    const pauseDuration = 100; // Pause for 5 seconds

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
        const hiddenInputValue = document.getElementById('savedMessagesHidden').value;
        const parsedValue = JSON.parse(hiddenInputValue);
        //  console.log("Parsed Value for function:", parsedValue);

        const bestAction = getBestActionFromModel(parsedValue); // Replace with your AI logic
        console.log("Best Action:", bestAction);

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


