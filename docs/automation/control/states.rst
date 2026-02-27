Flow and File States
=====================
The flow and file tables in the Reduction Database have a state property.
This is expected to change according to the following state diagrams.

File State Progression
---------------------------

.. mermaid::

    graph LR;
    planned --> creating;
    creating --> failed;
    creating --> created;
    creating --> unreported;
    created --> progressed;
    created --> quickpunched;
    quickpunched --> progressed;
    created --> levelh;
    levelh --> progressed;

For files, they always begin in the ``planned`` state.
Once the flow to create them is kicked off, they enter the ``creating`` state.
From ``creating`` they can either become ``failed`` (when the flow creating them
encounters an exception in execution),
``created`` (the flow succeeded in creating the file),
or ``unreported`` (when the flow that was supposed to update their state doesn't respond).
Once files are ``created`` they wait in that state until a later flow picks them up as ``progressed``
to the next level. Since level 3 is the last level, they never will become ``progressed``.

For Level 0 products, they advance to a ``levelh`` state from ``created``. This extra intermediate state
allows them to be processed into a calibration format that is used for deriving pipeline parameters.

For Level 1 products, they advanced to a ``quickpunched`` state from ``created``.
This extra intermediate state allows the faster production of QuickPUNCH products before
they're used to make level 2 images.


Flow State Progression
---------------------------

.. mermaid::

    graph LR;
    planned --> launched;
    launched --> running;
    running --> failed;
    running --> completed;

Flow behavior is a bit simpler. Flows are ``planned`` by the scheduler.
The launcher changes them to ``launched`` when they're submitted to Prefect.
When the flow begins running it updates its status to ``running``.
From ``running`` they can either enter the ``failed`` or ``completed`` states depending
on the success of their execution.
