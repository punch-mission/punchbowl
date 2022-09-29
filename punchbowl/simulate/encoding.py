import numpy as np
import os.path

TABLE_PATH = os.path.dirname(__file__)


class Encoding:
    @staticmethod
    def encode(
        data: np.ndarray = np.zeros([2048, 2048]), frombits: int = 16, tobits: int = 12
    ) -> np.ndarray:
        """
        Square root encode between specified bitrate values

        Parameters
        ----------
        data
            Input data array (n x n)
        frombits
            Specified bitrate of original input image
        tobits
            Specifed bitrate of output encoded image

        Returns
        -------
        np.ndarray
            Encoded version of input data (n x n)

        """

        data = np.round(data).astype(np.ulonglong).clip(0, None)
        ibits = tobits * 2
        factor = np.array(2 ** (ibits - frombits)).astype(np.ulonglong)
        s2 = (data * factor).astype(np.ulonglong)

        return np.round(np.sqrt(s2)).astype(np.ulonglong)

    @staticmethod
    def gen_decode_table(
        bias_level: float = 100,
        gain: float = 4.3,
        readnoise_level: float = 17,
        frombits: int = 12,
        tobits: int = 16,
    ) -> None:
        """
        Generates a square root decode table between specified bitrate values and CCD parameters

        Parameters
        ----------
        bias_level
            ccd bias level
        gain
            ccd gain
        readnoise_level
            ccd read noise level
        frombits
            Specified bitrate of encoded image to unpack
        tobits
            Specified bitrate of output data (decoded)

        Returns
        -------
        None

        """

        int_type = np.int32

        def encode(source, frombits, tobits):
            source = np.round(source).astype(int_type).clip(0, None)
            ibits = (
                tobits * 2
            )  # Intermediate step is multiplication to exactly 2x target bit value
            factor = np.array(2 ** (ibits - frombits))  # force a 1-element numpy array
            s2 = np.round(source * factor).astype(
                int_type
            )  # source*factor is normally an int anyhow but force integer arithmetic
            return np.round(np.sqrt(s2)).astype(
                int_type
            )  # force nearest-integer square root

        def decode(source, frombits, tobits):
            source = (
                np.round(source).astype(int_type).clip(0, None)
            )  # Force integer arithmetic and nonnegative values
            ibits = tobits * 2  # Calculate factor as in encode above.
            factor = 2.0 ** (ibits - frombits)
            s2 = source * source  # Square the rounded square root
            return np.round(s2 / factor).astype(
                int_type
            )  # nearest-integer division of the square

        def noise_pdf(val, gain, offset, fixedsigma, n_sigma=5, n_steps=10000):
            electrons = np.clip(
                (val - offset) / gain, 1, None
            )  # Use camera calibration to get an e- count
            poisson_sigma = (
                np.sqrt(electrons) * gain
            )  # Shot noise, converted back to DN
            sigma = np.sqrt(
                poisson_sigma**2 + fixedsigma**2
            )  # Total sigma is quadrature sum of fixed & shot
            step = (
                sigma * n_sigma * 2 / n_steps
            )  # Step through a range in DN value -n_sigma to n_sigma in n_steps
            dn_steps = np.arange(
                -n_sigma * sigma, n_sigma * sigma, step
            )  # Explicitly enumerate the step values
            normal = np.exp(
                -dn_steps * dn_steps / sigma / sigma / 2
            )  # Explicitly calculate the Gaussian/normal PDF at each step
            normal = normal / np.sum(
                normal
            )  # Easier to normalize numerically than to account for missing tails
            return (val + dn_steps, normal)

        def mean_b_offset(sval, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma):
            val = decode(sval, frombits, tobits)  # Find the "naive" decoded value
            (vals, weights) = noise_pdf(
                val, ccd_gain, ccd_offset, ccd_fixedsigma
            )  # Generate a distribution around that naive value
            weights = weights * (
                vals >= ccd_offset
            )  # Ignore values below the offset -- which break the noise model
            if (
                np.sum(weights) < 0.95
            ):  # At or below the ccd offset, just return no delta at all.
                return 0
            weights = weights / np.sum(weights)
            svals = encode(
                vals, frombits, tobits
            )  # Encode the entire value distribution
            dcvals = decode(
                svals, frombits, tobits
            )  # Decode the entire value distribution to find the net offset
            ev = np.sum(dcvals * weights)  # Expected value of the entire distribution
            return ev - val  # Return ΔB.

        def decode_corrected(
            sval, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma
        ):
            s1p = decode(sval + 1, frombits, tobits)
            s1n = decode(sval - 1, frombits, tobits)
            width = (s1p - s1n) / 4
            fixed_sigma = np.sqrt(ccd_fixedsigma**2 + width**2)
            of = mean_b_offset(
                sval, frombits, tobits, ccd_gain, ccd_offset, fixed_sigma
            )
            return decode(sval, frombits, tobits) - of

        def gen_decode_table(frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma):
            output = np.zeros(2**tobits)
            for i in range(0, 2**tobits):
                output[i] = decode_corrected(
                    i, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma
                )
            return output

        table = gen_decode_table(frombits, tobits, gain, bias_level, readnoise_level)

        # TODO - FITS output

        filename = (
            TABLE_PATH
            + "/decoding_tables/"
            + "tab_fb"
            + str(frombits)
            + "_tb"
            + str(tobits)
            + "_g"
            + str(gain)
            + "_b"
            + str(bias_level)
            + "_r"
            + str(readnoise_level)
            + ".npy"
        )

        np.save(filename, table)

    @staticmethod
    def decode(
        data: np.ndarray = np.zeros([2048, 2048]),
        bias_level: float = 100,
        gain: float = 4.3,
        readnoise_level: float = 17,
        frombits: int = 12,
        tobits: int = 16,
    ) -> np.ndarray:
        """
        Square root decode between specified bitrate values

        Parameters
        ----------
        data
            input encoded data array (n x n)
        bias_level
            ccd bias level
        gain
            ccd gain
        readnoise_level
            ccd read noise level
        frombits
            Specified bitrate of encoded image to unpack
        tobits
            Specified bitrate of output data (decoded)

        Returns
        -------
        np.ndarray
            square root decoded version of the input image (n x n)

        """

        int_type = np.int32

        def decode_bytable(s, table):
            s = np.round(s).astype(int_type).clip(0, table.shape[0])
            return table[s]

        tablename = (
            TABLE_PATH
            + "/decoding_tables/"
            + "tab_fb"
            + str(frombits)
            + "_tb"
            + str(tobits)
            + "_g"
            + str(gain)
            + "_b"
            + str(bias_level)
            + "_r"
            + str(readnoise_level)
            + ".npy"
        )

        # Check for an existing table, otherwise generate one
        if os.path.isfile(tablename):
            table = np.load(tablename)
        else:
            Encoding.gen_decode_table(
                bias_level, gain, readnoise_level, frombits, tobits
            )
            table = np.load(tablename)

        return decode_bytable(data, table)
