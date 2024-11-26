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

    // Add touchstart and touchmove event listeners to the canvas
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);

    function handleTouch(event) {
        event.preventDefault(); // Prevent default touch behavior (e.g., scrolling)

        const touch = event.touches[0]; // Get the first touch point
        const canvasRect = canvas.getBoundingClientRect(); // Get canvas dimensions and position
        const touchX = touch.clientX - canvasRect.left; // X-coordinate relative to canvas
        const touchY = touch.clientY - canvasRect.top; // Y-coordinate relative to canvas

        // Determine the center of the canvas
        const centerX = canvasRect.width / 2;

        // Apply force based on touch position
        if (touchX < centerX) {
            const message = 'Touch detected: Left side - Applying left force';
            console.log(message);
            updateOutputMessage(message); // Update index page
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: -0.05, y: 0 });
        } else {
            const message = 'Touch detected: Right side - Applying right force';
            console.log(message);
            updateOutputMessage(message); // Update index page  
            Matter.Body.applyForce(lowerArm, lowerArm.position, { x: 0.05, y: 0 });
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

// Function to add keyboard control
function addKeyboardControl(pendulum) {
    const lowerArm = pendulum.bodies[1];
    document.addEventListener('keydown', (event) => {
        console.log(`Key pressed: ${event.key}`);
        const key = event.key;
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
        }
    });
}

// Draw Zone Indicators on the Grid
function drawZoneLabels(context, canvasWidth, canvasHeight) {
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;

    context.font = '16px Arial';
    context.fillStyle = '#555';

    // Top-Left
    context.fillText('Top-Left', centerX / 2 - 30, centerY / 2);

    // Top-Right
    context.fillText('Top-Right', centerX + centerX / 2 - 40, centerY / 2);

    // Bottom-Left
    context.fillText('Bottom-Left', centerX / 2 - 40, centerY + centerY / 2);

    // Bottom-Right
    context.fillText('Bottom-Right', centerX + centerX / 2 - 50, centerY + centerY / 2);
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

Simulation.doublePendulum = (containerId, centerX, centerY) => {
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
    };

    // Create runner
    const runner = Runner.create();
    Runner.run(runner, engine);

    // Add bodies
    const group = Body.nextGroup(true);
    const length = 200;
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
    
    Composites.chain(pendulum, 0.45, 0, -0.45, 0, { 
        stiffness: 0.9, 
        length: 0,
        angularStiffness: 0.7,
        render: {
            strokeStyle: '#4a485b'
        }
    });

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

    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];

    // Rotate the lower arm 90 degrees (Math.PI/2) around its position
    Body.rotate(lowerArm, Math.PI/2, {
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

    Events.on(render, 'afterRender', () => {
        // Draw the grid
        drawGrid(render.context, render.options.width, render.options.height);

        // Draw the pendulum trail
        trail.unshift({
            position: Vector.clone(lowerArm.position),
            speed: lowerArm.speed
        });

        Render.startViewTransform(render);
        render.context.globalAlpha = 0.7;

        trail.forEach((pointInfo) => {
            const { position: point, speed } = pointInfo;
            const hue = 250 + Math.round((1 - Math.min(1, speed / 10)) * 170);
            render.context.fillStyle = `hsl(${hue}, 100%, 55%)`;
            render.context.fillRect(point.x, point.y, 2, 2);
        });

        render.context.globalAlpha = 1;
        Render.endViewTransform(render);

        if (trail.length > 2000) {
            trail.pop();
        }

        // Output the y-value of the pendulum's endpoint
        //console.log("Current Y position of lower arm endpoint:", lowerArm.position.y);
        
        // Calculate the y position of the pendulum's endpoint
        const yEndpoint = calculateYEndpoint(lowerArm, pendulum.bodies[0], length, length);
        
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

    // keep the mouse in sync with rendering
    render.mouse = mouse;

    // fit the render viewport to the scene
    Render.lookAt(render, {
        min: { x: 0, y: 0 },
        max: { x: container.clientWidth, y: container.clientHeight }
    });

    // Call the function to add keyboard control
    addKeyboardControl(pendulum);

    addTouchControl(pendulum, render.canvas);

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


