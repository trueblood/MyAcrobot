var Simulation = Simulation || {};

Simulation.doublePendulum = (containerId) => {
    const { Engine, Events, Render, Runner, Body, Composite, Composites, Constraint, MouseConstraint, Mouse, Bodies, Vector } = Matter;

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
    
    Composite.add(pendulum, Constraint.create({ 
        bodyB: pendulum.bodies[0],
        pointB: { x: -length * 0.42, y: 0 },
        pointA: { x: pendulum.bodies[0].position.x - length * 0.42, y: pendulum.bodies[0].position.y },
        stiffness: 0.9,
        length: 0,
        render: {
            strokeStyle: '#4a485b'
        }
    }));

    const lowerArm = pendulum.bodies[1];
    const upperArm = pendulum.bodies[0];

    // Set the pendulum to a stable, vertical position with no initial movement
    Body.setPosition(lowerArm, { x: 300, y: 160 + length });
    Body.setAngle(lowerArm, 0);
    Body.setVelocity(lowerArm, { x: 0, y: 0 });
    Body.setAngularVelocity(lowerArm, 0);
    
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
    });

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

    render.mouse = mouse;

    Render.lookAt(render, {
        min: { x: 0, y: 0 },
        max: { x: container.clientWidth, y: container.clientHeight }
    });

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
