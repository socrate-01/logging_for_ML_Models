from typing import List, Optional
import logging
from ml_base.decorator import MLModelDecorator


class LoggingDecorator(MLModelDecorator):
    """Decorator for logging around an MLModel instance."""

    def __init__(self, input_fields: Optional[List[str]] = None,
                 output_fields: Optional[List[str]] = None) -> None:
        super().__init__(input_fields=input_fields, output_fields=output_fields)
        self.__dict__["_logger"] = None

    def predict(self, data):
        if self.__dict__["_logger"] is None:
            self.__dict__["_logger"] = logging.getLogger("{}_{}".format(
                self._model.qualified_name, "logger"))

        # extra fields to be added to the log record
        extra = {
            "action": "predict",
            "model_qualified_name": self._model.qualified_name,
            "model_version": self._model.version
        }

        # adding model input fields to the extra fields to be logged
        new_extra = dict(extra)
        if self._configuration["input_fields"] is not None:
            for input_field in self._configuration["input_fields"]:
                new_extra[input_field] = getattr(data, input_field)

        self.__dict__["_logger"].info("Prediction requested.", extra=new_extra)

        try:
            prediction = self._model.predict(data=data)
            extra["status"] = "success"

            # adding model output fields to the extra fields to be logged
            new_extra = dict(extra)
            if self._configuration["output_fields"] is not None:
                for output_field in self._configuration["output_fields"]:
                    new_extra[output_field] = getattr(prediction, output_field)
            self.__dict__["_logger"].info("Prediction created.", extra=new_extra)
            return prediction
        except Exception as e:
            extra["status"] = "error"
            extra["error_info"] = str(e)
            self.__dict__["_logger"].error("Prediction exception.", extra=extra)
            raise e
