Pipeline Design
===============

This data flow is orchestrated using the `Prefect <https://github.com/PrefectHQ/prefect>`_ framework, which handles and monitors the pipeline as a whole. This is bundled into a PUNCH specific processing tool named `punchpipe <https://github.com/punch-mission/punchpipe>`_. Within punchbowl, individual flows are defined for each data level, which outline the series of pipeline modules that the data will pass through.

The code snippet below provides an example for a punchbowl level 3 flow. This flow is marked with a @flow decorator, and takes in a list of data to process through the marked modules. Using the punchbowl code, an end user could use this as a starting point for bespoke processing - either modifying an existing flow to skip processing steps, or defining a new processing flow altogether.

.. code-block:: python

    @flow(validate_parameters=False)
    def level3_core_flow(data_list: list[str] | list[NDCube],
                        before_f_corona_model_path: str,
                        after_f_corona_model_path: str,
                        starfield_background_path: str | None,
                        output_filename: str | None = None) -> list[NDCube]:
        """Level 3 core flow."""
        logger = get_run_logger()

        logger.info("beginning level 3 core flow")
        data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
        data_list = [subtract_f_corona_background_task(d,
                                                    before_f_corona_model_path,
                                                    after_f_corona_model_path) for d in data_list]
        data_list = [subtract_starfield_background_task(d, starfield_background_path) for d in data_list]
        data_list = [convert_polarization(d) for d in data_list]
        logger.info("ending level 3 core flow")

        if output_filename is not None:
            output_image_task(data_list[0], output_filename)

        return data_list


For individual pipeline modules, such as the example F-corona subtraction below, the @punch_task decorator is used to mark this as a punch specific task, orchestrated as part of a data flow as defined above. This task handles the overall data flow and logging, with the primary processing happening inside a separate function.

.. code-block:: python

    @punch_task
    def subtract_f_corona_background_task(observation: NDCube,
                                         before_f_background_model_path: str,
                                         after_f_background_model_path: str) -> NDCube:
        """
        Subtracts a background f corona model from an observation.

        This algorithm linearly interpolates between the before and after models.

        Parameters
        ----------
        observation : NDCube
            an observation to subtract an f corona model from

        before_f_background_model_path : str
            path to a NDCube f corona background map before the observation

        after_f_background_model_path : str
            path to a NDCube f corona background map after the observation

        Returns
        -------
        NDCube
            A background subtracted data frame

        """
        logger = get_run_logger()
        logger.info("subtract_f_corona_background started")


        before_f_corona_model = load_ndcube_from_fits(before_f_background_model_path)
        after_f_corona_model = load_ndcube_from_fits(after_f_background_model_path)

        output = subtract_f_corona_background(observation, before_f_corona_model, after_f_corona_model)
        output.meta.history.add_now("LEVEL3-subtract_f_corona_background", "subtracted f corona background")

        logger.info("subtract_f_corona_background finished")

        return output

As specified above, the above task handles logging and pipeline orchestration, with a standalone function defined as below handling the primary reduction step. This function can therefore be extracted / remixed for use inside a custom processing pipeline.

.. code-block:: python

    def subtract_f_corona_background(data_object: NDCube,
                                    before_f_background_model: NDCube,
                                    after_f_background_model: NDCube ) -> NDCube:
        """Subtract f corona background."""
        # check dimensions match
        if data_object.data.shape != before_f_background_model.data.shape:
            msg = (
                "f_background_subtraction expects the data_object and"
                "f_background arrays to have the same dimensions."
                f"data_array dims: {data_object.data.shape} "
                f"and before_f_background_model dims: {before_f_background_model.data.shape}"
            )
            raise InvalidDataError(
                msg,
            )

        if data_object.data.shape != after_f_background_model.data.shape:
            msg = (
                "f_background_subtraction expects the data_object and"
                "f_background arrays to have the same dimensions."
                f"data_array dims: {data_object.data.shape} "
                f"and after_f_background_model dims: {after_f_background_model.data.shape}"
            )
            raise InvalidDataError(
                msg,
            )

        before_date = before_f_background_model.meta.datetime.timestamp()
        after_date = after_f_background_model.meta.datetime.timestamp()
        observation_date = data_object.meta.datetime.timestamp()

        if before_date > observation_date:
            msg = "Before F corona model was after the observation date"
            raise InvalidDataError(msg)

        if after_date < observation_date:
            msg = "After F corona model was before the observation date"
            raise InvalidDataError(msg)

        if before_date == observation_date:
            interpolated_model = before_f_background_model
        elif after_date == observation_date:
            interpolated_model = after_f_background_model
        else:
            interpolated_model = ((after_f_background_model.data - before_f_background_model.data)
                                * (observation_date - before_date) / (after_date - before_date)
                                + before_f_background_model.data)

        interpolated_model[np.isinf(data_object.uncertainty.array)] = 0

        data_object.data[...] = data_object.data[...] - interpolated_model
        data_object.uncertainty.array[:, :] -= interpolated_model
        return data_object
