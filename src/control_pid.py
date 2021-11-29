class ControlPID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        """ Define the Gains for PID controller

        :param kp: Gain for proportional part
        :param ki: Gain for integral part
        :param kd: Gain for derivative part
        :param dt: Time step
        """
        self._step_time = 0.0
        self._last_step = 0.0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._past_error = 0.0
        self._error = 0.0
        self._sum_error = 0.0
        self.co = 0.0

    def compute(self, actual_value, set_point, step_time, filter_time=0.0, filter_co=False):
        """ Compute the PID controller.

        :param filter_co: Active the CO filter
        :param actual_value: Actual value
        :param set_point: The set point for the PID
        :param step_time: Step time for discrete PID
        :param filter_time: Filter time for CO filter
        :param debug: Print the information about the PID and error
        """

        # Compute error
        self._step_time = step_time
        self._error = round(actual_value - set_point, 5)
        self._sum_error = round((self._error + self._sum_error) * self._step_time, 5)

        # Proportional error
        p_output = round(self.kp * self._error, 5)
        # Integral error
        i_output = round(self.ki * self._sum_error * self._step_time, 5)
        # Derivative error
        if self._step_time <= 0.0:
            d_output = round(self.kd * ((self._error - self._past_error) / self._last_step), 5)

        else:
            d_output = round(self.kd * ((self._error - self._past_error) / self._step_time), 5)
            self._last_step = self._step_time

        self._past_error = self._error

        output = round(p_output + i_output + d_output, 5)

        # CO Filter
        if filter_co:
            self.co = self.co + ((self._step_time / filter_time) * (output - self.co))
        else:
            self.co = output

        return self.co
