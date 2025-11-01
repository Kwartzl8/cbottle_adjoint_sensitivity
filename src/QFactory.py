from dataclasses import dataclass
from cbottle.datasets.base import BatchInfo
import earth2grid
import numpy as np
import torch
from cbottle.datasets.dataset_2d import SST_MEAN, SST_SCALE

@dataclass
class QStruct:
    """Stores information about a quantity of interest and its gradient wrt SST."""
    name: str
    description: str | None
    channels_used: list[str]
    extent: tuple[float, float, float, float] | None  # (min_long, max_long, min_lat, max_lat) or None for global
    hpx_level: int
    num_pixels: int
    aggr_func: callable

class QFactory:
    """
    Factory for creating aggregation functions, scaling parameters and names for quantities of interest whose gradient wrt SST we want.
    """
    def __init__(
        self,
        batch_info: BatchInfo,
        hpx_level: int = 6,
    ):
        self.channels = batch_info.channels
        self.scale = batch_info.scales
        self.mean = batch_info.center
        self.hpx_level = hpx_level
        self.diff_var_center = SST_MEAN
        self.diff_var_scale = SST_SCALE


    def get_gradient_scaling(self) -> float:
        """
        Returns the gradient scaling factor for dq/dc.
        Since q is unnormalized in aggregation functions, this is simply 1/SST_SCALE.
        """
        return 1.0 / self.diff_var_scale
    
    # def get_pressure_velocity_500hPa(
    #         self,
    # ) -> QStruct:
    #     """
    #     Returns a QStruct for the global average of upward pressure velocity omega+ at 500hPa.
    #     We will calculate omega_500 in the following way:
    #     omega_500 = w_500
    #     Returns:
    #         QStruct containing all information about this quantity of interest.
    #     """
        

    def get_global_TOA_outgoing_radiation(
            self,
    ) -> QStruct:
        """
        Returns a QStruct for the global TOA outgoing radiation (rsut + rlut).
        Returns:
            QStruct containing all information about this quantity of interest.
        """
        rsut_scale, rsut_center = self.scale[self.channels.index("rsut")], self.mean[self.channels.index("rsut")]
        rlut_scale, rlut_center = self.scale[self.channels.index("rlut")], self.mean[self.channels.index("rlut")]
        
        def aggregation_fn(x):
            # denormalize and add them together - returns unnormalized q
            rsut = x[:, self.channels.index("rsut"), :, :] * rsut_scale + rsut_center
            rlut = x[:, self.channels.index("rlut"), :, :] * rlut_scale + rlut_center
            return (rsut + rlut).mean(dim=(-1,)) # no renormalization
        
        # Calculate total number of pixels for global average
        hpx_grid = earth2grid.healpix.Grid(level=self.hpx_level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
        total_pixels = hpx_grid._npix()
        
        return QStruct(
            name="global_TOA_outgoing_radiation",
            description="Global average of top-of-atmosphere outgoing radiation (rsut + rlut)",
            channels_used=["rsut", "rlut"],
            extent=None,  # Global
            hpx_level=self.hpx_level,
            num_pixels=total_pixels,
            aggr_func=aggregation_fn
        )
        
    def get_global_avg_in_channel(
        self, channel: str,
    ) -> QStruct:
        """
        Returns a QStruct for the global average of the given channel.
        Args:
            channel: The channel to average.
        Returns:
            QStruct containing all information about this quantity of interest.
        """
        assert channel in self.channels, f"Channel {channel} not found in {self.channels}."
        
        channel_scale = self.scale[self.channels.index(channel)]
        channel_center = self.mean[self.channels.index(channel)]
        
        def aggregation_fn(x):
            # Return unnormalized q
            normalized_channel = x[:, self.channels.index(channel), :, :]
            return (normalized_channel * channel_scale + channel_center).mean(dim=(-1,))

        # Calculate total number of pixels for global average
        hpx_grid = earth2grid.healpix.Grid(level=self.hpx_level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
        total_pixels = hpx_grid._npix()
        
        return QStruct(
            name=f"global_avg_{channel}",
            description=f"Global average of {channel}",
            channels_used=[channel],
            extent=None,  # Global
            hpx_level=self.hpx_level,
            num_pixels=total_pixels,
            aggr_func=aggregation_fn
        )


    def _slice_patch_indices(self, min_long, max_long, min_lat, max_lat):
        """
        Returns the indices of the patch in the healpix grid.
        """
        # Make the longitude increment smaller than the approximate HealPix longitude increment at the equator.
        hpx_grid = earth2grid.healpix.Grid(level=self.hpx_level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
        ground_resolution = hpx_grid.approximate_grid_length_meters() / 1000.0  # in km
        d_long = ground_resolution / (2* np.pi * 6371.0) * 360.0  # Convert to degrees
        d_long /= 1.9 # Make it smaller by an arbitrary constant to make sure we get all pixels in the patch.
        d_lat = d_long
        longs = torch.arange(min_long, max_long + d_long, d_long)
        lats = torch.arange(min_lat, max_lat + d_lat, d_lat)
        hpx_indices = hpx_grid.ang2pix(
            longs.unsqueeze(0).repeat(lats.shape[0], 1).flatten(),
            lats.unsqueeze(1).repeat(1, longs.shape[0]).flatten(),
        )

        return hpx_indices.flatten().unique()


    def get_patch_avg_in_channel(
        self, channel: str, min_long: float, max_long: float, min_lat: float, max_lat: float
    ) -> QStruct:
        """
        Returns a QStruct for the average of the given channel in a patch defined by the given coordinates.
        Args:
            channel: The channel to average.
            min_long: Minimum longitude of the patch.
            max_long: Maximum longitude of the patch.
            min_lat: Minimum latitude of the patch.
            max_lat: Maximum latitude of the patch.
        Returns:
            QStruct containing all information about this quantity of interest.
        """
        assert channel in self.channels, f"Channel {channel} not found in {self.channels}."
        hpx_indices_in_patch = self._slice_patch_indices(
            min_long, max_long, min_lat, max_lat
        )
        
        channel_scale = self.scale[self.channels.index(channel)]
        channel_center = self.mean[self.channels.index(channel)]

        def aggregation_fn(x):
            # Return unnormalized q
            normalized_channel = x[:, self.channels.index(channel), :, hpx_indices_in_patch]
            return (normalized_channel * channel_scale + channel_center).mean(dim=(-1,))
        
        return QStruct(
            name=f"patch_avg_{channel}_{min_long}_{max_long}_{min_lat}_{max_lat}",
            description=f"Average of {channel} in patch [{min_long}, {max_long}] x [{min_lat}, {max_lat}]",
            channels_used=[channel],
            extent=(min_long, max_long, min_lat, max_lat),
            hpx_level=self.hpx_level,
            num_pixels=len(hpx_indices_in_patch),
            aggr_func=aggregation_fn
        )
    
    def get_patch_avg_in_channels(
        self, channels: list[str], min_long: float, max_long: float, min_lat: float, max_lat: float
    ) -> list[QStruct]:
        """
        Returns a list of QStructs for the average of the given channels in a patch defined by the given coordinates.
        Args:
            channels: The channels to average.
            min_long: Minimum longitude of the patch.
            max_long: Maximum longitude of the patch.
            min_lat: Minimum latitude of the patch.
            max_lat: Maximum latitude of the patch.
        Returns:
            A list of QStructs containing all information about these quantities of interest.
        """
        return [self.get_patch_avg_in_channel(c, min_long, max_long, min_lat, max_lat) for c in channels]